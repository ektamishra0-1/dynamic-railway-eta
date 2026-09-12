# 🚆 Dynamic ETA Forecasting for Coaching Trains

> SIH26028 — Smart Automation

An intelligent railway ETA forecasting system that dynamically updates downstream train arrival predictions based on observed delays.

## �� Problem

Traditional railway ETA systems can struggle to dynamically propagate delays through a train's remaining journey.

This project aims to build a dynamic forecasting engine that:

- predicts future station delays
- propagates current delays downstream
- provides prediction uncertainty
- explains why ETA predictions changed
- supports historical journey replay as a simulation of live railway data

## 🧠 Approach

The system is being developed progressively:

1. Data validation and forensic analysis
2. Historical delay baselines
3. Time-aware feature engineering
4. ML-based delay forecasting
5. Dynamic downstream prediction
6. Uncertainty estimation
7. Explainability
8. Interactive railway control-room dashboard
9. FastAPI prediction service

## 🖥️ Dashboard

The frontend is being built with:

- React
- Vite
- Framer Motion
- Recharts
- Lucide React
- Tailwind CSS

The dashboard will provide an interactive railway control-room experience with historical journey replay and dynamic delay propagation.

## 📊 Current Dataset

Current local dataset:

- 8 trains
- 2,451 journeys
- 50,598 observations
- Historical station-level delay observations

The dataset is currently used locally and is not committed to Git.

## 🏗️ Project Structure

```text
SIH_train/
├── data/
│   ├── raw/
│   ├── processed/
│   └── visualizations/
│
├── src/
│   ├── 01_build_canonical.py
│   ├── 02_investigate_data.py
│   ├── 03_clean_data.py
│   ├── 04_eda.py
│   └── 05_visualizations.py
│
├── dashboard/
│   ├── src/
│   ├── public/
│   └── package.json
│
├── .gitignore
└── README.md
