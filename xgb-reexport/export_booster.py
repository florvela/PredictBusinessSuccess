# export_booster.py
import joblib
from pathlib import Path

# Ahora miramos en /app/models, no PredictBusinessSuccess/models
MODEL_PATH = Path("models/xgb_over.pickle")
OUT_PATH   = Path("xgb_over.json")

print(f"📥 Cargando modelo desde {MODEL_PATH}…")
m = joblib.load(MODEL_PATH)

print("🔄 Extrayendo Booster y guardando a JSON…")
booster = m.get_booster()
booster.save_model(str(OUT_PATH))

print(f"✅ Booster escrito en {OUT_PATH}")
