import sys
import os
import json
import torch
import numpy as np
import logging
import traceback  # Hataları tam görmek için
from kafka import KafkaConsumer
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

print("✅ 1. Kütüphaneler yüklendi.")

# --- PATH AYARLAMALARI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
mobility_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(mobility_dir)
sys.path.append(src_dir)

print(f"✅ 2. Path ayarlandı: {src_dir}")

try:
    from mobility.models import STGCN_LSTM
    from mobility.preprocessing import load_traffic_data, MinMaxNormalizer
    from mobility.utils.graph import load_adj_matrix, calculate_scaled_laplacian
    print("✅ 3. Proje modülleri (models, preprocessing) yüklendi.")
except ImportError as e:
    print(f"❌ MODÜL HATASI: {e}")
    sys.exit(1)

# LOGGING
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger("TrafficBrain")

# --- KONFİGÜRASYON ---
KAFKA_TOPIC = "traffic_sensor_data"
KAFKA_GROUP = "traffic_predictor_group"
BOOTSTRAP_SERVERS = ['localhost:9092']

INFLUX_URL = "http://localhost:8086"
# BURASI ÖNEMLİ: Docker'daki token ile burası aynı olmalı!
INFLUX_TOKEN = "my-super-secret-auth-token"
INFLUX_ORG = "my-org"
INFLUX_BUCKET = "traffic_data"

# Dosya Yolları
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODEL_PATH = os.path.join(BASE_DIR, "best_model_stgcn.pth")
DATA_PATH = os.path.join(src_dir, "mobility", "data", "metr-la.h5")
ADJ_PATH = os.path.join(src_dir, "mobility", "data", "adj_METR-LA.pkl")
SEQ_LEN = 12

class TrafficConsumer:
    def __init__(self):
        print("⚙️ Consumer başlatılıyor...")
        self.device = torch.device("cpu")
        
        # 1. Normalizer
        if not os.path.exists(DATA_PATH):
             print(f"❌ HATA: Veri dosyası bulunamadı: {DATA_PATH}")
             sys.exit(1)
        
        print("📊 Veri seti yükleniyor...")
        df_full = load_traffic_data(DATA_PATH)
        self.scaler = MinMaxNormalizer()
        self.scaler.fit(df_full.values)
        
        # 2. Graf Yapısı
        if not os.path.exists(ADJ_PATH):
            print(f"❌ HATA: Adj dosyası bulunamadı: {ADJ_PATH}")
            sys.exit(1)
            
        print("🕸️ Graf yapısı yükleniyor...")
        _, adj_mx = load_adj_matrix(ADJ_PATH)
        self.laplacian = calculate_scaled_laplacian(adj_mx).to(self.device)
        
        # 3. Model
        print(f"🧠 Model yükleniyor: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            print(f"❌ HATA: Model dosyası yok! {MODEL_PATH}")
            sys.exit(1)

        num_nodes = df_full.shape[1]
        self.model = STGCN_LSTM(num_nodes=num_nodes, in_features=1, hidden_dim=64, out_dim=1, K=3)
        
        try:
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            print("✅ Model ağırlıkları yüklendi.")
        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            sys.exit(1)

        self.model.to(self.device)
        self.model.eval()
        
        # 4. InfluxDB
        print("💾 InfluxDB'ye bağlanılıyor...")
        self.influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
        
        self.buffer = []
        print("✅ Consumer kurulumu tamamlandı.")

    def process_stream(self):
        print("🎧 Kafka bağlantısı deneniyor...")
        try:
            consumer = KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=BOOTSTRAP_SERVERS,
                auto_offset_reset='latest',
                enable_auto_commit=True,
                group_id=KAFKA_GROUP,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                consumer_timeout_ms=10000  # 10 saniye veri gelmezse hata verip çıkmasın diye
            )
            print("✅ Kafka'ya bağlanıldı. Mesaj bekleniyor...")
        except Exception as e:
            print(f"❌ Kafka Bağlantı Hatası: {e}")
            print("👉 İPUCU: Docker çalışıyor mu? 'docker ps' komutunu dene.")
            return

        for message in consumer:
            # Mesaj geldiğinde buraya girer
            data = message.value
            if 'sensor_readings' not in data:
                continue

            sensor_readings = np.array(data['sensor_readings'])
            
            self.buffer.append(sensor_readings)
            
            if len(self.buffer) > SEQ_LEN:
                self.buffer.pop(0)
            
            if len(self.buffer) == SEQ_LEN:
                self.predict_and_store(current_actual=sensor_readings)
            else:
                # Buffer dolana kadar bilgi verelim
                if len(self.buffer) % 5 == 0:
                    print(f"⏳ Buffer doluyor: {len(self.buffer)}/{SEQ_LEN}")

    def predict_and_store(self, current_actual):
        try:
            input_data = np.array(self.buffer)
            input_norm = self.scaler.transform(input_data)
            input_tensor = torch.FloatTensor(input_norm).unsqueeze(0).unsqueeze(-1).to(self.device)
            
            with torch.no_grad():
                prediction_norm = self.model(input_tensor, self.laplacian)
                
            prediction_norm = prediction_norm.cpu().numpy().squeeze()
            prediction_real = self.scaler.inverse_transform(prediction_norm)
            
            mae = np.mean(np.abs(prediction_real - current_actual))
            
            points = []
            
            # Tüm sensörleri kaydet
            for node_idx in range(len(current_actual)):
                p = Point("traffic_monitor") \
                    .field("actual_speed", float(current_actual[node_idx])) \
                    .field("predicted_speed", float(prediction_real[node_idx])) \
                    .tag("sensor_id", f"sensor_{node_idx}")
                points.append(p)

            p_mae = Point("traffic_monitor") \
                .field("network_mae", float(mae)) \
                .tag("sensor_id", "global_network")
            points.append(p_mae)
                
            self.write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=points)
            logger.info(f"🔮 Global MAE: {mae:.2f} | Sensor_0 Act: {current_actual[0]:.2f} vs Pred: {prediction_real[0]:.2f}")
            
        except Exception as e:
            print(f"❌ Tahmin/Yazma Hatası: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Script Başlatıldı (Main Block)")
    try:
        consumer = TrafficConsumer()
        consumer.process_stream()
    except KeyboardInterrupt:
        print("\n🛑 Kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\n❌ BEKLENMEYEN KRİTİK HATA:")
        traceback.print_exc()