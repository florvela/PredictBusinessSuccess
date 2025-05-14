FROM python:3.7-slim

# Install BLAS/LAPACK y los paquetes Debian de NumPy/SciPy
RUN apt-get update && apt-get install -y \
      build-essential \
      libopenblas-dev \
      liblapack-dev \
      python3-numpy \
      python3-scipy \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copio el modelo y el script
COPY models/xgb_over.pickle models/xgb_over.pickle
COPY xgb-reexport/export_booster.py export_booster.py

RUN pip install --upgrade pip \
 && pip install --no-cache-dir \
      joblib==0.14.0 \
      xgboost==0.90

RUN pip install --no-build-isolation --no-cache-dir "Cython<3"

# Instalo scikit-learn usando el numpy de apt y sin aislamiento
RUN pip install --no-build-isolation --no-cache-dir scikit-learn==0.21.3

CMD ["python", "export_booster.py"]
