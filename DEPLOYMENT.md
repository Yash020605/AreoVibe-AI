# 🚀 AeroVibe AI - Deployment Guide

AeroVibe AI is fully containerized and production-ready. The application has been optimized to run securely and consistently across any environment using Docker.

## Prerequisites
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## 1. Local / Edge Deployment (Drone Base Station)

If you are running the application on a ruggedized laptop or edge server at the farm (where the drone base station is located):

1. Open your terminal in the project directory.
2. Build and start the container in detached mode:
   ```bash
   docker-compose up -d --build
   ```
3. The application will be accessible via any device on the local network by navigating to:
   ```text
   http://<HOST_MACHINE_IP>:8501
   ```
   *(You can find the exact IP under the "Data Intake" section in the app sidebar).*

## 2. Cloud Deployment (AWS / GCP / DigitalOcean)

To deploy AeroVibe AI as a centralized web platform accessible from anywhere:

1. Provision a basic Linux VPS (e.g., AWS EC2 t3.medium or DigitalOcean Droplet with at least 4GB RAM).
2. SSH into your server, clone the repository, and run the standard Docker Compose command:
   ```bash
   docker-compose up -d --build
   ```
3. Ensure that your cloud firewall (Security Groups in AWS) allows incoming TCP traffic on Port `8501`.
4. (Optional) Set up an Nginx reverse proxy to map Port `8501` to Port `80` (HTTP) or `443` (HTTPS with Let's Encrypt), allowing you to bind a custom domain like `app.aerovibe.ai`.

> [!TIP]
> **Performance Scaling**
> The current architecture utilizes ONNX Runtime CPU. If you deploy to a GPU-enabled instance (e.g., AWS g4dn), you can install the `nvidia-docker2` toolkit to drastically increase throughput, allowing multiple concurrent live drone streams!
