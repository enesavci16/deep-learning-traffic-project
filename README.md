```
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

We trained and compared a baseline LSTM model against the proposed ST-GCN model. The Graph Neural Network significantly outperformed the baseline by capturing the spatial connectivity between roads.

| Model | Val Loss (MSE) | Notes |
| :--- | :--- | :--- |
| **Baseline (LSTM)** | 0.0213 | Learns only temporal patterns. |
| **ST-GCN (Proposed)** | **0.0085** | **60% Improvement.** Learns spatio-temporal dependencies. |

> **⚠️ Note on Metrics:** The model is trained on normalized data (scaled between 0 and 1). The ST-GCN model demonstrates superior convergence and generalization capabilities compared to standard LSTM.

## 🚀 Getting Started

### Prerequisites
* **Docker Desktop** (running)
* **Python 3.10+**
* **uv** (recommended for package management) or `pip`

### 1. Installation & Setup
Clone the repository and install dependencies:

```bash
# Using uv (Recommended)
uv sync

# Or using standard pip
pip install -r requirements.txt


2. Data Preparation (Graph Generation)
Before training, generate the adjacency matrix (Graph structure) from the traffic data to avoid data leakage:

Bash

uv run python src/mobility/utils/generate_adj.py
This creates adj_METR-LA.pkl in the data folder.

3. Model Training (Reproduce Results)
You can train both models using the unified trainer script.

Train Baseline (LSTM):

Bash

uv run python src/mobility/train.py --data src/mobility/data/metr-la.h5 --model lstm --epochs 10
Train Proposed Model (ST-GCN):

Bash

uv run python src/mobility/train.py --data src/mobility/data/metr-la.h5 --model stgcn --adj src/mobility/data/adj_METR-LA.pkl --epochs 10
4. Evaluation (Benchmark)
To generate MAE and RMSE comparison tables on the test set:

Bash

uv run python src/mobility/evaluate.py --data src/mobility/data/metr-la.h5 --adj src/mobility/data/adj_METR-LA.pkl
5. Run the Real-Time Pipeline (Streaming)
Once you have trained the models, you can deploy them in the real-time pipeline.

Step A: Start Infrastructure

Bash

docker-compose up -d
Step B: Start the Consumer (The Brain)

Bash

uv run python src/mobility/pipeline/consumer.py
Step C: Start the Producer (The Data Source)

Bash

uv run python src/mobility/pipeline/producer.py
6. Monitor the Results
Open your browser to http://localhost:8086.

Login with user: admin / password: password123.

Go to Explore and select the traffic_data bucket.

Filter by measurement traffic_monitor to see the live graph.

📂 Project Structure
Plaintext

├── docker-compose.yml       # Infrastructure orchestration
├── src/
│   └── mobility/
│       ├── data/            # Dataset (.h5) and Graph (.pkl)
│       ├── models/          # 
│       │   ├── core.py      # ST-GCN and LSTM PyTorch definitions
│       │   └── __init__.py
│       ├── pipeline/        # Kafka Producer and Consumer scripts
│       ├── preprocessing.py # Data loading and normalization logic
│       ├── train.py         # Main training loop
│       ├── evaluate.py      # Testing and benchmarking script
│       └── utils/           # Graph generation utilities
├── best_model_lstm.pth      # Saved model weights
├── best_model_stgcn.pth     # Saved model weights
└── README.md                # Project documentation
🛠️ Technologies Used
Language: Python 3.10

ML Frameworks: PyTorch (ST-GCN, LSTM)

Streaming: Apache Kafka, AIOKafka

Database: InfluxDB (Time Series)

Containerization: Docker
```
