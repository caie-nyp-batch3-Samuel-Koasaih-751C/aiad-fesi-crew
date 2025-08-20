# AIAD FESI Crew Project

This project implements a full machine learning workflow using **Kedro**, **Docker**, and **Kubernetes** with a Flask UI for inference.  
It covers data ingestion, preprocessing, masking, dataset splitting, model training, and a production-ready UI served via Gunicorn behind Kubernetes with autoscaling.

---

## Project Structure

aiad-fesi-crew/
├── conf/ # Kedro configs (catalog, parameters, etc.)
├── data/ # Local data (ignored in Docker/K8s)
├── docker/ # Dockerfiles and entrypoints
│ ├── Dockerfile.kedro
│ ├── Dockerfile.train
│ ├── Dockerfile.ui
│ ├── entrypoint.sh
│ ├── entrypoint.train.sh
│ ├── gunicorn.conf.py
│ └── requirements.ui.txt
├── k8s/ # Kubernetes manifests
│ ├── cm-kedro-conf.yaml
│ ├── job-ingestion.yaml
│ ├── job-data-preprocessing.yaml
│ ├── job-mask-apply.yaml
│ ├── job-split.yaml
│ ├── job-train.yaml
│ ├── pvc-data.yaml
│ ├── deploy-ui.yaml
│ ├── ui-service.yaml
│ ├── ui-hpa.yaml
│ └── ui-ingress.yaml
├── notebooks/ # Jupyter notebooks (EDA, prototyping)
├── src/aiad_fesi_crew/ # Main Kedro + UI source
│ ├── pipelines/ # Kedro pipelines
│ │ ├── data_ingestion/
│ │ ├── data_preprocessing/
│ │ ├── mask_merge/
│ │ ├── data_split/
│ │ └── training/
│ └── ui/ # Flask UI
│ ├── static/
│ ├── templates/
│ ├── init.py # create_app()
│ ├── routes.py
│ └── inference.py
├── docker-compose.yml # Local multi-service runner
├── pyproject.toml # Dependencies
├── requirements.txt # Python deps
└── README.md # You are here


---

## Running Kedro Pipelines Locally

1. Install dependencies:

```bash
pip install -r requirements.txt

    Run a pipeline:

kedro run --pipeline data_split
kedro run --pipeline training

Pipelines available:

    data_ingestion

    data_preprocessing

    mask_merge

    data_split

    training

Running with Docker
Build Images

# Kedro worker image
docker build -f docker/Dockerfile.kedro -t aiad-fesi-crew-kedro:v5 .

# Training image
docker build -f docker/Dockerfile.train -t aiad-fesi-crew-train:v3 .

# UI image
docker build -f docker/Dockerfile.ui -t aiad-fesi-crew-ui:v3 .

Run UI Locally

docker run --rm -p 8000:8000 aiad-fesi-crew-ui:v3

Then open http://localhost:8000

.
Running on Kubernetes (Minikube)
Start Minikube

minikube start

Deploy Data PVC

kubectl apply -f k8s/pvc-data.yaml

Run Data/Training Jobs

kubectl apply -f k8s/job-ingestion.yaml
kubectl apply -f k8s/job-data-preprocessing.yaml
kubectl apply -f k8s/job-mask-apply.yaml
kubectl apply -f k8s/job-split.yaml
kubectl apply -f k8s/job-train.yaml

Deploy UI

kubectl apply -f k8s/deploy-ui.yaml
kubectl apply -f k8s/ui-service.yaml

(Optional) Horizontal Pod Autoscaler

kubectl apply -f k8s/ui-hpa.yaml

Ingress Access

    Enable ingress:

minikube addons enable ingress

Apply ingress:

kubectl apply -f k8s/ui-ingress.yaml

Update /etc/hosts:

192.168.49.2   ui.potatoe

(replace with your minikube ip)

Open http://ui.potatoe
⚡ Load Testing HPA

To simulate load and test scaling:

# Install hey (Arch)
sudo pacman -S hey

# Run load test
hey -z 30s -c 50 http://ui.potatoe/

Monitor scaling:

kubectl get hpa -w

Features Implemented

    Kedro pipelines for data processing and training

    Dockerized pipelines and UI

    Kubernetes jobs for each pipeline stage

    PVC for persistent dataset sharing

    UI deployment with health probes

    Service + Ingress for external access

    Horizontal Pod Autoscaler (HPA)

    Load testing setup with hey

Demo Script (Quick Run)

# 1. Start cluster
minikube start

# 2. Deploy PVC
kubectl apply -f k8s/pvc-data.yaml

# 3. Run preprocessing + training
kubectl apply -f k8s/job-ingestion.yaml
kubectl apply -f k8s/job-data-preprocessing.yaml
kubectl apply -f k8s/job-mask-apply.yaml
kubectl apply -f k8s/job-split.yaml
kubectl apply -f k8s/job-train.yaml

# 4. Deploy UI
kubectl apply -f k8s/deploy-ui.yaml
kubectl apply -f k8s/ui-service.yaml
kubectl apply -f k8s/ui-ingress.yaml
kubectl apply -f k8s/ui-hpa.yaml

# 5. Access UI
minikube ip   # check cluster IP
# add to /etc/hosts
# <IP>   ui.potatoe

👨‍💻 Authors: AIAD FESI Crew
📅 Version: August 2025