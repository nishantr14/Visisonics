# OTShield — AI-Powered OT Anomaly Detection

OTShield is a multi-modal anomaly detection system for Operational Technology (OT) / Industrial Control Systems (ICS). It uses an ensemble ML pipeline (Random Forest + Gradient Boosting) trained on the BATADAL water treatment SCADA dataset to detect cyberattacks in real time.

## Architecture

```
Modbus/PLC  -->  Data Collector  -->  Supervised Scorer (RF+GBM)
                                          |
                                     Cyber Layer  +  Physical Layer
                                          |
                                     Fusion Engine  -->  Risk Level
                                          |
                                   FastAPI + WebSocket
                                          |
                                   Live Dashboard (Chart.js)
```

**Detection layers:**
- **Cyber Layer** — Network/command anomaly scoring via ensemble model
- **Physical Layer** — Sensor telemetry deviation scoring
- **Audio Layer** — Acoustic anomaly detection (planned)
- **Visual Layer** — Camera-based detection (planned)
- **Fusion Engine** — Weighted multi-modal risk aggregation

## Performance

| Metric | OTShield | Zahoor et al. 2025 (Baseline) |
|--------|----------|-------------------------------|
| CV F1 (5-fold) | **0.795** | 0.835 |
| CV AUC | **0.967** | 0.891 |
| Features | 71 (43 sensors + 28 engineered) | — |
| Model | Ensemble RF + GBM | — |

## Quick Start

### 1. Install dependencies

```bash
cd otshield
pip install -r requirements.txt
```

### 2. (Optional) Retrain the model

```bash
python train_supervised.py
```

This trains the ensemble on BATADAL datasets 03 (normal) and 04 (attacks), runs stratified 5-fold CV, and saves model artifacts to `models/`.

### 3. Start the server

```bash
cd src
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open the dashboard

- Landing page: [http://localhost:8000](http://localhost:8000)
- Live dashboard: [http://localhost:8000/dashboard](http://localhost:8000/dashboard)

The dashboard streams real-time predictions via WebSocket every 2 seconds. Use the scenario buttons (Normal / Cyber Attack / Physical Fault / Critical) to simulate different attack modes.

## Project Structure

```
otshield/
  data/                  # BATADAL CSV datasets
  frontend/
    index.html           # Landing page
    dashboard.html       # Live monitoring dashboard
    hero.mp4             # Hero video
  models/                # Trained model artifacts (.pkl, .json)
  notebooks/
    01_cyber_layer.ipynb
    02_physical_layer.ipynb
    03_benchmark_and_evaluation.ipynb
  src/
    api.py               # FastAPI backend + WebSocket
    api_models.py         # Pydantic schemas
    supervised_scorer.py  # Inference module
    cyber_layer.py        # Cyber scoring wrapper
    physical_layer.py     # Physical scoring wrapper
    fusion.py             # Multi-modal fusion engine
    fake_plc_stream.py    # Simulated PLC data stream
    data_collector.py     # Real Modbus/PLC data collector
  train_supervised.py     # Model training script
  requirements.txt
```

## Tech Stack

- **ML**: scikit-learn (RandomForest + GradientBoosting ensemble), pandas, numpy
- **Backend**: FastAPI, WebSocket, uvicorn
- **Frontend**: Vanilla JS, Chart.js
- **OT Integration**: pymodbus (Modbus TCP), OpenPLC compatible
