import sys
import os

# --- PATH FIX START ---
# This allows Python to find the 'mobility' package inside 'src'
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up 2 levels: 'pipeline' -> 'mobility' -> 'src'
src_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(src_dir)
# --- PATH FIX END ---

import logging
import asyncio
import pandas as pd
import json
from datetime import datetime, timedelta  # <--- NEW IMPORT for live simulation
from pathlib import Path
from typing import Optional, Dict
from aiokafka import AIOKafkaProducer

# Import your Pydantic model
from mobility.schemas.validation import TrafficSnapshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class TrafficProducer:
    def __init__(self, bootstrap_servers: str, topic: str):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.producer: Optional[AIOKafkaProducer] = None

    async def start(self):
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        await self.producer.start()
        logger.info("Kafka Producer started.")

    async def stop(self):
        if self.producer:
            await self.producer.stop()

    async def stream_data(self, df: pd.DataFrame, interval: float = 0.5):
        logger.info(f"Streaming {len(df)} time steps...")
        
        # 1. Get the current time as the starting point (RIGHT NOW)
        current_base_time = datetime.utcnow()
        
        # Iterating by Time Step (Row)
        # We enumerate to calculate the time offset
        for i, (original_timestamp, row) in enumerate(df.iterrows()):
            try:
                # 2. Create a FAKE "Now" timestamp
                # Each step adds 5 minutes to the current time to simulate a live feed
                # So row 1 is Now, row 2 is Now + 5 mins, etc.
                simulated_time = current_base_time + timedelta(seconds=10*i)

                # 3. Convert row to dictionary {sensor_id: speed}
                readings = {str(k): float(v) for k, v in row.to_dict().items()}
                
                # 4. Create Snapshot with SIMULATED live time
                snapshot = TrafficSnapshot(
                    timestamp=simulated_time.isoformat(),
                    sensor_readings=readings
                )
                
                # 5. Send message
                await self.producer.send(
                    self.topic, 
                    snapshot.model_dump(mode='json')
                )
                
                logger.info(f"Sent snapshot for {simulated_time}")
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error sending snapshot: {e}")

async def main():
    KAFKA_SERVER = "localhost:9092"
    TOPIC = "traffic_live_v1"

    # --- DATA PATH FIX ---
    DATA_DIR = os.path.join(src_dir, 'mobility/data')
    FILE_PATH = os.path.join(DATA_DIR, 'metr-la.h5')

    if not os.path.exists(FILE_PATH):
        logger.error(f"Data file not found at: {FILE_PATH}")
        return

    # Quick Load
    logger.info(f"Loading data from {FILE_PATH}...")
    df = pd.read_hdf(FILE_PATH)
    
    # Use first 2000 rows for demo
    df = df.iloc[:2000]

    producer = TrafficProducer(KAFKA_SERVER, TOPIC)
    await producer.start()
    try:
        await producer.stream_data(df)
    finally:
        await producer.stop()

if __name__ == "__main__":
    asyncio.run(main())