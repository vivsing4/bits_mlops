# Kubernetes Deployment

Apply all manifests:

```bash
kubectl apply -f deploy/k8s-manifests
```

Port-forward (local verification):

```bash
kubectl port-forward svc/heart-disease-api 8000:80
curl http://localhost:8000/health
```

The deployment includes Prometheus scrape annotations for `/metrics`.
