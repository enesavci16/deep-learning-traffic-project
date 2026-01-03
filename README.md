# 🚦 Real-Time Urban Traffic Prediction (End-to-End Pipeline)

## 📖 Project Overview

This project is a **full-stack data engineering and machine learning pipeline** designed to predict urban traffic speeds in real-time. It leverages **Spatio-Temporal Graph Convolutional Networks (ST-GCN)** to model the complex spatial and temporal dependencies of traffic flow.

The system ingests streaming sensor data via **Apache Kafka**, processes it using a Deep Learning consumer, and visualizes live "Actual vs. Predicted" accuracy on an **InfluxDB** dashboard.

## 🏗️ Architecture

The pipeline is containerized using **Docker** and consists of four main stages:

1.  **Data Ingestion (Producer):** A Python service that streams historical METR-LA data, simulating a live environment by injecting real-time UTC timestamps.
2.  **Message Broker (Kafka):** Buffers and transports high-throughput sensor data asynchronously.
3.  **Inference Engine (Consumer):**
    * Subscribes to Kafka topics.
    * Maintains a sliding window of time-series data.
    * Runs inference using a trained **ST-GCN** model.
    * Calculates **MAE (Mean Absolute Error)** in real-time.
4.  **Visualization (InfluxDB):** Stores and plots the raw traffic speed, predicted speed, and error rates for live monitoring.

## 🧠 Model Performance

We compared a baseline LSTM model against the ST-GCN model. The Graph Neural Network significantly outperformed the baseline by capturing the spatial connectivity between roads.

| Model | Normalized MAE (Loss) | Real-World MAE (km/h) | Notes |
| :--- | :--- | :--- | :--- |
| **Baseline (LSTM)** | 0.0468 | ~5.85 km/h | Learns only temporal patterns. |
| **ST-GCN (Proposed)** | **0.0158** | **2.92 km/h** | Learns spatio-temporal dependencies. |

> **⚠️ Note on Metrics:** The model is trained on normalized data (scaled between 0 and 1). The loss value `0.0158` represents this normalized error. When denormalized back to actual traffic speeds, our model achieves an **MAE of 2.92 km/h**. This result is highly competitive with state-of-the-art models like DCRNN (Li et al., 2018), which reported an MAE of 2.77.

## 🚀 Getting Started

### Prerequisites
* **Docker Desktop** (running)
* **Python 3.10+**
* **uv** (recommended) or `pip`

### 1. Start the Infrastructure
Spin up Zookeeper, Kafka, and InfluxDB using Docker Compose:
```bash
docker-compose up -d

2. Install Dependencies

# If using uv (recommended)
uv sync

# Or standard pip
pip install -r requirements.txt


3. Run the Real-Time Pipeline
Open two terminal windows:

Terminal A: Start the Consumer (The Brain)
uv run python src/mobility/pipeline/consumer.py

Terminal B: Start the Producer (The Data Source)
uv run python src/mobility/pipeline/producer.py

. Monitor the Results
Open your browser to http://localhost:8086.

Login with user: admin / password: password123.

Go to Explore and select the traffic_data bucket.

Filter by measurement traffic_monitor to see the live graph.

📂 Project Structure
├── docker-compose.yml       # Infrastructure orchestration
├── src/
│   ├── mobility/
│   │   ├── models/          # ST-GCN and LSTM PyTorch definitions
│   │   ├── pipeline/        # Producer and Consumer scripts
│   │   ├── preprocessing/   # Adjacency matrix generation
│   │   └── schemas/         # Data validation (Pydantic)
├── models/                  # Saved .pth model weights
└── README.md                # Project documentation


🛠️ Technologies Used
Language: Python

ML Frameworks: PyTorch, PyTorch Geometric

Streaming: Apache Kafka, AIOKafka

Database: InfluxDB (Time Series)

Containerization: Docker

