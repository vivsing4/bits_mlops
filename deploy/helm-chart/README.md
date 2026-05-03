# Helm Chart

Install locally:

```bash
helm upgrade --install heart-disease-api deploy/helm-chart
```

Override image:

```bash
helm upgrade --install heart-disease-api deploy/helm-chart \
  --set image.repository=<registry>/heart-disease-api \
  --set image.tag=<tag>
```
