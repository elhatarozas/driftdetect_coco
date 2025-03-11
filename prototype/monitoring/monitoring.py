# Copyright 2021 The KServe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Union
import os 
from kserve import (
    Model,
    ModelServer,
    model_server,
    InferRequest,
    InferResponse,
    InferOutput,
    logging
)
from kserve.model_server import app


import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models 
import torchdrift
from io import BytesIO
import base64
from PIL import Image
import time
import asyncio
import queue
import threading 
from prometheus_client import Gauge, REGISTRY

class Monitoring(Model):
    def __init__(self, name: str):
        super().__init__(name, return_response_headers=True)
        self.model = None
        self.load()

    def load(self):
        loader = torch.utils.data.DataLoader(datasets.CIFAR10(root="/mnt/models/cifar10", download=False, transform=transforms.ToTensor()))

        collected = 0
        l = []
        for sample in loader:
            if sample[1] != 1:
                continue 
            if collected >= 1000:
                break 
            collected += 1
            l.append(sample[0])
        training_tensor = torch.cat(l, axis=0)
        self.detector = ImageDrift(training_data=training_tensor)
        try:
            self.drift_detected = Gauge("drift_detected", "Binäre Variable zur Feststellung, ob ein Data Drift erkannt worden ist")
            self.enough_data = Gauge("enough_data", "Binäre Variable zur Feststellung, ob genug Daten für eine Drifterkennung vorliegen")
            self.mmd = Gauge("mmd", "Maximum Mean Distance Metrik zur Feststellung von Modelldrift")
            self.p_value = Gauge("p_value", "p-Wert des Hypothesentests zur Feststellung von Modelldrift via Permutationstest")
        except ValueError:
            self.drift_detected = REGISTRY._names_to_collectors["drift_detected"]
            self.enough_data = REGISTRY._names_to_collectors["enough_data"]
            self.mmd = REGISTRY._names_to_collectors["mmd"]
            self.p_value = REGISTRY._names_to_collectors["p_value"]
        
        self.queue = queue.Queue()
        self.dd_thread = threading.Thread(target=self.drift_detect_loop, daemon=True)
        self.dd_thread.start()
        self.ready = True

    async def predict(
        self,
        payload: Union[Dict, InferRequest],
        headers: Dict[str, str] = None,
        response_headers: Dict[str, str] = None,
    ) -> Union[Dict, InferResponse]:
        
        # Add your monitoring Logic here .. ++
        logging.logger.info(payload)

        image_in = transforms.ToTensor()(Image.open(BytesIO(base64.b64decode(payload.inputs[0].data[0]))))
        self.queue.put_nowait(image_in)
        print(f"Queue length{self.queue.qsize()}")
        return InferResponse(model_name='monitoring',response_id='1', infer_outputs=[InferOutput(name="output-0", shape=[3], datatype="UINT32" ,data=[3, 1, 2])])
    
    def drift_detect_loop(self):
        while True:
            logging.logger.info("Awaiting a new element from the queue...")
            input = self.queue.get()
            self.detector.add_data(input)
            result = self.detector.predict()
            logging.logger.info(result)
            self.enough_data.set(result["enough-data"]*1) # bool to int
            self.drift_detected.set(result["drift-detected"]*1)
            self.p_value.set(result["p-value"])
            self.mmd.set(result["mmd"])

parser = argparse.ArgumentParser(parents=[model_server.parser])
args, _ = parser.parse_known_args()

class ImageDrift:

    def __init__(self, training_data, *, window_size=50, dimred_size=10, preprocessing="default", dimred="none", p_value_threshhold=.05):
        self.dimred : str = dimred
        self.dimred_size=dimred_size
        self.window_size = window_size
        self.p_value_threshhold : float = p_value_threshhold
        self.preprocess = self.init_preprocess_training(preprocessing)
        self.init_training_data(training_data)
        self.init_transform()
        self.testdata = torch.empty([0] + list(self.tensors.shape[1:]))
        

    def init_preprocess_training(self, preprocessing):
        return transforms.Compose([ # transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    def init_training_data(self, training_data):
        tensors = self.preprocess(training_data)
        if "resnet" in self.dimred:
            self.resnet = models.resnet50()
            self.resnet.load_state_dict(torch.load("/mnt/models/resnet50.pt"))
            self.resnet.eval()
            self.tensors = self.resnet(tensors)
        else:
            self.tensors = torch.flatten(tensors, start_dim=1)
        if "pca" in self.dimred:
            self.pca = torchdrift.reducers.PCAReducer(self.dimred_size)
            self.pca.fit(self.tensors)
            self.tensors = self.pca(self.tensors)
        self.drift_detect = torchdrift.detectors.KernelMMDDriftDetector(return_p_value=False)
        self.drift_detect.fit(self.tensors)

    def init_transform(self):
        # Expecting data in Tensorform
        operators = []
        if "resnet" in self.dimred:
            operators.append(self.resnet)
        else:
            operators.append(nn.Flatten())
        
        if "pca" in self.dimred:
            operators.append(self.pca)
        # operators.append(self.drift_detect)
        self.transform= nn.Sequential(*operators)

    def add_data(self, data):
        data = self.preprocess(data)
        self.testdata = torch.cat([self.testdata, self.transform(torch.unsqueeze(data, 0))])
        if self.testdata.shape[0] > self.window_size:
            self.testdata = self.testdata[1:, ...]

    def predict(self):
        result = {}
        if self.testdata.shape[0] < self.window_size:
            result["drift-detected"] = False
            result["enough-data"] = False
            result["p-value"] = -1
            result["mmd"] = -1
            return result
        result["enough-data"] = True
        a = time.time()
        result["mmd"] = self.drift_detect(self.testdata).item()
        result["p-value"] = self.drift_detect.compute_p_value(self.testdata).item()
        logging.logger.info(f"Zeit für Drift Detection:{time.time() - a}")
        result["drift-detected"] = result["p-value"]  < self.p_value_threshhold
        return result

if __name__ == "__main__":
    if args.configure_logging:
        logging.configure_logging(args.log_config_file)
    model = Monitoring(args.model_name)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    ModelServer().start([model])
