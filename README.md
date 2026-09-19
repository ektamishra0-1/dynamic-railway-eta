# 🚆 RailPulse AI — Dynamic ETA Forecasting for Coaching Trains

> **Smart India Hackathon 2026 (SIH26028) — Smart Automation**  
> Dynamic railway delay propagation and expected time of arrival (ETA) intelligence powered by PyTorch GRU + Temporal Attention.

---

## 📌 Problem Overview

Traditional railway ETA systems rely on static timetables and point-to-point average speeds. When disruptions occur, these systems struggle to dynamically propagate delay momentum through downstream sections:

- **Rigid Timetables**: Delays are often assumed to vanish or persist uniformly without modeling section congestion or train priority.
- **Compounding Bottlenecks**: A 15-minute delay on an upstream trunk line frequently triggers cascading dispatch delays across subsequent junctions.
- **Lack of Uncertainty & Explainability**: Passengers and railway controllers are provided with single-point arrival estimates with no confidence bounds or explanatory rationale.

**RailPulse AI** solves this by modeling sequential delay evolution as a dynamic time-series problem. Using historical telemetry across key Indian Railways corridors, it predicts incremental delay changes ($\Delta$) station-by-station and auto-regressively updates downstream ETAs in real time.

---

## 🧠 AI Architecture: GRU + Temporal Attention

The primary forecasting engine uses a custom **2-Layer PyTorch GRU with Additive Temporal Attention** (`models/gru_attention_best.pt`):

```
                        [ Categorical Features ]          [ Numeric Features ]
                        (Train No, Station, Prev)         (Delays, Deltas, Time)
                                   │                                 │
                            Entity Embeddings                        │
                              (3 x 16-dim)                           │
                                   └──────────────┬──────────────────┘
                                                  ▼
                                       Concatenated Input (dim=56)
                                                  │
                                         2-Layer GRU (dim=128)
                                        (Seq Len = 6 Stations)
                                                  │
                                         Temporal Attention
                                    (Tanh-Weighted Context Vector)
                                                  │
                                           Dropout (0.25)
                                                  │
                                            MLP Regressor
                                                  │
                                                  ▼
                                 Next Station Delay Change (Δ in min)
```

- **Sequence Lookback**: 6 consecutive historical station traversals ($L = 6$).
- **Features**:
  - **8 Numeric Features**: `journey_station_index`, `delay`, `previous_delay`, `delay_change`, `day_of_week`, `day_of_month`, `month`, `is_weekend`.
  - **3 Categorical Embeddings**: `train_no` (vocab 9 $\to$ 16d), `station_code` (vocab 161 $\to$ 16d), `previous_station` (vocab 156 $\to$ 16d).
- **Target Variable**: Next station delay change ($\Delta = \text{delay}_{t+1} - \text{delay}_t$).
- **Total Parameters**: **192,354 weights**.
- **Downstream Rollout**: For multi-horizon forecasts (+1 to +4 stations), the engine runs recursive auto-regressive rollout with a $1/\sqrt{h}$ stability damping factor.

---

## 📊 Model Performance & Baseline Benchmark

The model has been independently audited and validated on 4,713 held-out supervised test sequences (July 12, 2026 – September 5, 2026):

| Model / Strategy | MAE (Delay Change) | RMSE (Delay Change) | MAE vs. Persistence |
| :--- | :--- | :--- | :--- |
| **Persistence Baseline ($\Delta = 0$)** | 18.51 min | 30.43 min | 0.00% (Baseline) |
| **Train Historical Median** | 18.54 min | 30.47 min | -0.16% |
| **Section Median** | 14.12 min | 25.76 min | +23.74% |
| **GRU + Attention (RailPulse)** | **12.62 min** | **22.71 min** | **+31.85% Improvement** |

### Tolerance Coverage (Absolute Error)
- **Within $\pm 5$ min**: **39.91%**
- **Within $\pm 10$ min**: **62.27%**
- **Within $\pm 15$ min**: **75.60%**
- **Within $\pm 30$ min**: **91.11%**
- **Median Absolute Error**: **6.75 minutes**

### Per-Train Performance Highlights
- **12952 (Mumbai Rajdhani Express)**: **4.39 min MAE** (highest priority dispatch corridor).
- **12958 (Swarna Jayanti Rajdhani)**: **5.71 min MAE** ($R^2 = 0.778$).
- **12628 (Karnataka Express)**: **10.98 min MAE** across 34 cross-country stations.
- Detailed audit logs and findings are documented in [`reports/gru_attention_evaluation.md`](reports/gru_attention_evaluation.md).

---

## 🖥️ Interactive RailPulse Dashboard

Built with **React 19**, **Vite**, **Recharts**, **Lucide Icons**, and custom responsive styling:

1. **Live Journey Simulation**:
   - Replays historical train journeys station-by-station with simulated telemetry (speed, current delay, section status).
   - Interactive Route Stepper with real-time train icon animation along track milestones.
   - Controls: Play, Pause, Resume, Reset, Train Selector, and Date Picker.
2. **Four Specialized Views**:
   - **Journey View**: Interactive route position, station inspector, delay trend graph, and confidence bounds.
   - **Train Status View**: Telemetry cards (speed, section traversal, station index, delay volatility).
   - **Rail Network View**: Topological corridor visualization with station distance markers.
   - **AI Insights View**: 
     - **Downstream Delay & Risk Matrix**: Station-by-station predicted delay, delay change ($\Delta$), 95% confidence interval, and risk classification pill (`ON TIME`, `LOW DELAY IMPACT`, `MODERATE RISK`, `SEVERE DELAY`).
     - **Neural Architecture Blueprint**: 2-layer GRU, multi-head attention, and stability damping parameters.
     - **Attention Weights Attribution**: Feature importance breakdown for delay velocity, arrival anchor, and section clearance.
3. **What-If Simulation & Copilot**:
   - What-If delay slider to simulate upstream delays and observe immediate downstream impact.
   - RailPulse AI Assistant providing interactive natural language explanations of ETA adjustments.

---

## 🏗️ Project Architecture

```text
SIH_train/
├── backend/
│   └── app/
│       ├── forecasting/
│       │   ├── gru_model.py            # PyTorch GRUAttention neural architecture
│       │   ├── predictor.py            # Preprocessing & inference pipeline
│       │   └── gru_forecasting.py      # Production forecasting wrapper
│       ├── services/
│       │   ├── eta_engine.py           # Recursive downstream delay propagation
│       │   └── live_simulator.py       # Station-by-station journey replay engine
│       ├── main.py                     # FastAPI REST API endpoints
│       └── schemas.py                  # Pydantic request/response models
│
├── dashboard/                          # Frontend React application
│   ├── src/
│   │   ├── App.jsx                     # Master dashboard component
│   │   ├── App.css                     # Control-room styling & animations
│   │   └── main.jsx                    # React entrypoint
│   └── package.json
│
├── models/
│   ├── gru_attention_best.pt           # Trained PyTorch GRU checkpoint
│   ├── gru_attention_metrics.json      # Original training metrics
│   └── gru_attention_evaluation.json   # Reproducible audit results
│
├── reports/
│   └── gru_attention_evaluation.md     # Formal model audit & benchmark report
│
├── src/
│   ├── 06_prepare_model_data.py        # Sequence feature engineering pipeline
│   ├── 07_benchmark.py                 # Tabular baseline models (persistence, medians)
│   ├── 09_train_gru_attention.py       # Training pipeline with early stopping
│   └── evaluate_gru_attention.py       # Standalone reproducible evaluation script
│
├── data/
│   └── processed/                      # Preprocessed model sequences (CSV)
│
└── README.md
```

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm

### 1. Backend Setup & API Service
```bash
# Activate virtual environment
source .venv/bin/activate

# Install Python dependencies
pip install fastapi uvicorn torch pandas numpy scikit-learn

# Launch FastAPI development server
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
API documentation will be available at: `http://127.0.0.1:8000/docs`

### 2. Frontend Dashboard Setup
```bash
# In a separate terminal:
cd dashboard

# Install dependencies
npm install

# Start Vite development server
npm run dev -- --host 127.0.0.1 --port 5173
```
Access the dashboard at: `http://127.0.0.1:5173`

### 3. Run Reproducible Model Audit
To re-evaluate the GRU + Attention model against the held-out test set and verify all baseline comparisons:
```bash
./.venv/bin/python src/evaluate_gru_attention.py
```
This script evaluates Train, Validation, and Test sets, generates baseline comparisons, per-train breakdowns, tolerance coverages, and saves updated metrics to `models/gru_attention_evaluation.json`.

---

## 📡 Core API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/model/info` | Returns active neural network specifications and feature list |
| `GET` | `/trains` | Lists all 8 supported coaching trains |
| `GET` | `/dates/{train_no}` | Returns all available journey dates for a train |
| `GET` | `/route/{train_no}/{journey_date}` | Returns complete route with station sequence indices |
| `POST` | `/live/start/{train_no}/{journey_date}` | Initializes simulation replay at origin station |
| `POST` | `/live/tick/{train_no}/{journey_date}` | Advances the train to the next station |
| `GET` | `/live/forecast/{train_no}/{journey_date}` | Returns recursive downstream ETA forecasts with confidence intervals |

---

## 👥 Contributors & Acknowledgements

Developed for **Smart India Hackathon 2026** under Problem Statement **SIH26028: Dynamic Forecast of Expected Time of Arrival (ETA) for Coaching Trains**.
