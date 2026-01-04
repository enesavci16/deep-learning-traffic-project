import sys
import os
import time
import json
import numpy as np
import logging
from kafka import KafkaProducer
from datetime import datetime

# --- PATH AYARLAMALARI ---
# mobility paketini bulabilmesi için
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(src_dir)

from mobility.preprocessing import load_traffic_data

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrafficProducer")

# KAFKA AYARLARI
KAFKA_TOPIC = "traffic_sensor_data"
KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'

def json_serializer(data):
    return json.dumps(data).encode('utf-8')

def run_producer(data_path):
    # 1. Veriyi Yükle
    logger.info(f"⏳ Loading historical data from {data_path}...")
    try:
        df = load_traffic_data(data_path)
    except Exception as e:
        logger.error(f"Veri yüklenemedi: {e}")
        return

    # DataFrame'i Numpy array'e çevir
    data_values = df.values
    
    logger.info(f"✅ Data loaded. Shape: {data_values.shape}. Starting stream...")

    # 2. Kafka Producer Başlat
    try:
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
            value_serializer=json_serializer
        )
    except Exception as e:
        logger.error(f"Kafka'ya bağlanılamadı. Docker konteynerleri ayakta mı? Hata: {e}")
        return

    # 3. Veriyi Akıt (Streaming Loop)
    for i, row in enumerate(data_values):
        try:
            # Veriyi hazırla
            message = {
                "timestamp": datetime.utcnow().isoformat(),
                "step_index": i,
                "sensor_readings": row.tolist() # Numpy array seri hale gelmez, listeye çevir
            }
            
            # Kafka'ya gönder
            producer.send(KAFKA_TOPIC, message)
            
            # Logla (Her 100 adımda bir veya ilk adımlarda)
            if i % 100 == 0:
                logger.info(f"📤 Sent Step {i}/{len(data_values)} | Sensor[0] Speed: {row[0]:.2f}")
            
            # Simülasyon Hızı: Gerçek hayatta 5 dk olan arayı burada 0.5 saniye yapıyoruz
            time.sleep(0.5) 
            
        except KeyboardInterrupt:
            logger.info("🛑 Producer stopped by user.")
            break
        except Exception as e:
            logger.error(f"❌ Error sending message: {e}")

    producer.close()
    logger.info("🏁 Streaming finished.")

if __name__ == "__main__":
    # Veri dosyasının yolu
    data_file = os.path.join(src_dir, "mobility", "data", "metr-la.h5")
    
    if not os.path.exists(data_file):
        logger.error(f"File not found: {data_file}")
    else:
        run_producer(data_file)