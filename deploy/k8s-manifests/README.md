# Kubernetes Deployment

Apply all manifests:

```bash
kubectl apply -f deploy/k8s-manifests
```

Check resources:

```bash
kubectl get pods,svc,ingress
```

Port-forward (local verification):

```bash
kubectl port-forward svc/heart-disease-api 8000:80
curl http://localhost:8000/health
```

Prediction test:

```bash
curl -X POST http://localhost:8000/predict \
	-H "Content-Type: application/json" \
	-d '{
		"age": 58,
		"sex": 1,
		"cp": 2,
		"trestbps": 130,
		"chol": 250,
		"fbs": 0,
		"restecg": 1,
		"thalach": 140,
		"exang": 0,
		"oldpeak": 1.2,
		"slope": 2,
		"ca": 0,
		"thal": 2
	}'
```

Metrics endpoint verification:

```bash
curl http://localhost:8000/metrics
```

Ingress note:

- The provided Ingress uses host `heart-disease.local`.
- Add a local hosts entry if needed: `127.0.0.1 heart-disease.local`.

The deployment includes Prometheus scrape annotations for `/metrics`.
