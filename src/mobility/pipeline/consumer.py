import sys
import os

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '../../')) 
sys.path.append(src_dir)
# ----------------

import logging
import asyncio
import json
import torch
import numpy as np
from collections import deque
from aiokafka import AIOKafkaConsumer
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

from mobility.models.gnn_lstm import ST_GCN_LSTM
from mobility.preprocessing import load_adjacency_matrix, MinMaxNormalizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class TrafficConsumer:
    def __init__(self, bootstrap_servers, topic, model_path, influx_config, adj_path):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.window_size = 12
        self.data_buffer = deque(maxlen=self.window_size)
        
        # Resources
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.scaler = MinMaxNormalizer()
        self.scaler._min = 0.0; self.scaler._max = 70.0 
        
        # InfluxDB
        self.influx_client = InfluxDBClient(**influx_config)
        self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
        self.bucket = "traffic_data"
        self.sensor_order = None 

        # Load Graph
        adj = load_adjacency_matrix(adj_path)
        if adj is None:
            logger.warning(f"Adjacency matrix not found at {adj_path}! Using Identity (Low Accuracy).")
            self.laplacian = torch.eye(207).to(self.device)
        else:
            logger.info(f"Loaded Adjacency Matrix from {adj_path}")
            self.laplacian = torch.tensor(adj, dtype=torch.float32).to(self.device)

    def _load_model(self, path):
        model = ST_GCN_LSTM(num_nodes=207, in_channels=1, out_channels=1, lstm_units=64, K=2)
        if not os.path.exists(path):
            logger.error(f"Model not found at {path}")
            sys.exit(1)
            
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    async def start(self):
        self.consumer = AIOKafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            value_deserializer=lambda v: json.loads(v.decode('utf-8'))
        )
        await self.consumer.start()
        logger.info(f"Subscribed to topic: {self.topic}")

    async def stop(self):
        await self.consumer.stop()
        self.influx_client.close()

    def _prepare_input(self, readings_dict):
        if self.sensor_order is None:
            self.sensor_order = sorted(readings_dict.keys())
        vector = np.array([readings_dict[k] for k in self.sensor_order])
        vector = self.scaler.transform(vector)
        return vector.reshape(-1, 1)

    async def run_inference_loop(self):
        logger.info("Waiting for data stream...")
        async for msg in self.consumer:
            try:
                data = msg.value 
                timestamp = data['timestamp']
                readings = data['sensor_readings']
                
                # --- NEW: Calculate Actual Global Average for Monitoring ---
                real_vals = list(readings.values())
                actual_val = sum(real_vals) / len(real_vals)
                
                input_vector = self._prepare_input(readings)
                self.data_buffer.append(input_vector)

                if len(self.data_buffer) == self.window_size:
                    seq_np = np.array(self.data_buffer)
                    input_tensor = torch.tensor(seq_np, dtype=torch.float32).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        prediction = self.model(input_tensor, self.laplacian)
                        pred_val = self.scaler.inverse_transform(prediction.cpu().numpy())
                        
                        if pred_val.size == 1:
                            # Scalar Prediction (Global Avg)
                            pred_mph = pred_val.item()
                            # Calculate Error
                            mae_error = abs(pred_mph - actual_val)
                            
                            # Write Monitoring Data (Actual, Pred, Error)
                            self._write_monitor_data(timestamp, actual_val, pred_mph, mae_error)
                        else:
                            # Batch Prediction (Per Sensor)
                            self._write_batch_prediction(timestamp, pred_val.flatten())
                            
            except Exception as e:
                logger.error(f"Inference Error: {e}")

    def _write_monitor_data(self, timestamp, actual, predicted, error):
        """
        Writes comprehensive monitoring data to InfluxDB.
        """
        # Point 1: Actual
        p1 = Point("traffic_monitor").tag("type", "actual").field("speed", float(actual)).time(timestamp)
        # Point 2: Predicted
        p2 = Point("traffic_monitor").tag("type", "predicted").field("speed", float(predicted)).time(timestamp)
        # Point 3: Error
        p3 = Point("traffic_monitor").tag("type", "error").field("mae", float(error)).time(timestamp)

        try:
            self.write_api.write(bucket=self.bucket, record=[p1, p2, p3])
            # Log nice summary to console
            logger.info(f"Time: {timestamp} | Actual: {actual:.2f} | Pred: {predicted:.2f} | Diff: {error:.2f}")
        except Exception as e:
             logger.warning(f"InfluxDB Write Failed: {e}")

    def _write_batch_prediction(self, timestamp, values):
        points = []
        for i, val in enumerate(values):
            p = Point("traffic_prediction").tag("sensor_id", self.sensor_order[i]).field("speed", float(val)).time(timestamp)
            points.append(p)
        try:
            self.write_api.write(bucket=self.bucket, record=points)
            logger.info(f"Wrote {len(points)} sensor predictions.")
        except Exception as e:
            logger.warning(f"InfluxDB Write Failed: {e}")

async def main():
    KAFKA = "localhost:9092"
    TOPIC = "traffic_live_v1"
    
    # Absolute Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL = os.path.abspath(os.path.join(BASE_DIR, '../../../models/gnn_model.pth'))
    ADJ = os.path.abspath(os.path.join(src_dir, 'mobility/data/adj_METR-LA.pkl'))
    
    INFLUX = {"url": "http://localhost:8086", "token": "my-token", "org": "traffic_org"}

    consumer = TrafficConsumer(KAFKA, TOPIC, MODEL, INFLUX, ADJ)
    
    await consumer.start()
    try:
        await consumer.run_inference_loop()
    finally:
        await consumer.stop()

if __name__ == "__main__":
    asyncio.run(main())