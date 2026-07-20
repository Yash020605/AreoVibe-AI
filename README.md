# 🚁 AeroVibe AI

![AeroVibe Banner](https://img.shields.io/badge/AeroVibe-AI_Powered_Agriculture-00A859?style=for-the-badge) ![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge) ![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B?style=for-the-badge) ![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge)

AeroVibe AI is an advanced, autonomous Multi-Agent system designed for real-time agricultural disease detection using drone footage. By processing live video feeds directly from agricultural drones, AeroVibe maps crop health indices and detects early signs of pathologies, enabling precision agriculture at scale.

## ✨ Key Features

- **Multi-Agent Orchestration**: Powered by LangGraph, the system utilizes a specialized team of AI agents:
  - 🧠 *Supervisor Agent*: Orchestrates the data flow and checks for valid environmental conditions (e.g., lighting, leaf presence).
  - 👁️ *Vision Agent*: Processes raw imagery, runs heuristic computer vision engines, and generates scientific heatmaps.
  - 🔬 *Analysis Agent*: Diagnoses detected symptoms against a structured agricultural knowledge base to identify specific plant diseases.
  - 🎯 *Coordination Agent*: Synthesizes raw data into actionable treatment plans and alerts.
- **True VARI Heatmap Analysis**: Utilizes the Visible Atmospherically Resistant Index (VARI) mapped to a high-fidelity `TURBO` colormap to scientifically visualize plant stress and chlorophyll content without requiring multispectral infrared cameras.
- **Premium User Interface**: Built on Streamlit with injected Glassmorphism CSS, offering a responsive, dynamic, and breathtaking real-time dashboard for drone operators.
- **Edge & Cloud Ready**: Fully containerized using Docker, designed to run both locally on farm-edge servers and centrally in cloud environments (AWS/GCP).

## 🚀 Getting Started

### 1. Running Locally (Development)

Clone the repository and install the dependencies using `pip`:

```bash
git clone https://github.com/Yash020605/AreoVibe-AI.git
cd AreoVibe-AI
pip install -r requirements.txt
python -m streamlit run app.py
```

### 2. Running via Docker (Production / Edge)

AeroVibe AI is optimized for isolated container execution.

```bash
docker-compose up -d --build
```

Access the dashboard via `http://localhost:8501`. 
*For complete deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).*

## 🧠 Architecture Overview

The system is designed around a stateful pipeline (`state.py`). As imagery (or mock drone feeds) enters the system via `app.py`, the state is passed sequentially through the agents in the `agents/` directory.

- **`knowledge_base/`**: Contains `pathologies.json` mapping visual symptoms to crop diseases and recommended treatments.
- **`utils/`**: Contains core logic like the `image_fx.py` (VARI Heatmap Generator), `leaf_detector.py` (Vegetation confirmation via cascades and HSV tracking), and `heuristic_vision_engine.py` (Texture & color isolation).

## 🛠️ Built With
- **LangGraph** & **LangChain Core**: Agentic Workflow Routing
- **OpenCV** (Headless): Core Image Processing & Spatial Analysis
- **Streamlit**: Web Dashboard & UI
- **NumPy & Pandas**: Matrix Operations & Data Structuring

## 📝 License
This project is proprietary software developed for the AeroVibe AI initiative.
