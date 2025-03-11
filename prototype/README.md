1. Run "pack build --builder heroku/builder:24 jogalois/monitoring:v1" inside monitoring/
2. Run "docker push jogalois/monitoring:v1"
3. Deploy monitor-isvc.yaml
4. Deploy isvc.yaml
5. Send input.json to isvc.yaml 
```bash
curl -H "Content-Type: application/json" http://sklearn-iris.model.10.97.138.18.nip.io/v2/models/sklearn-iris/infer -d @./input.json
```
6. Check logs of monitor-isvc.yaml