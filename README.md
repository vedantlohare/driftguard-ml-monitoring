# 🛡️ DriftGuard: Real-Time ML Data Quality & Drift Monitoring Platform

[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Kafka](https://img.shields.io/badge/Apache%20Kafka-7.4.0-black.svg)](https://kafka.apache.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791.svg)](https://www.postgresql.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A production-grade, distributed observability and streaming platform designed to detect silent machine learning model degradation and data corruption in real-time.**

---

## 📌 Problem Overview

Machine learning models deployed in production fail silently. When upstream data schemas break or real-world consumer behavior drifts, traditional APM tools (like Datadog or New Relic) only see HTTP `200 OK` status codes with valid prediction floats.

**DriftGuard** acts as an automated statistical watchtower that:
1. **Intercepts streaming events** before inference via **Apache Kafka**.
2. **Performs deterministic schema & physical boundary checks** (e.g. invalid age ranges, missing fields, type errors).
3. **Calculates statistical distribution drift** on rolling windows using the **Kolmogorov-Smirnov (KS) Test** (for continuous features) and **Population Stability Index (PSI)** (for categorical features).
4. **Logs alerts and metrics** to a centralized **PostgreSQL** database.
5. **Serves real-time health metrics** via a **FastAPI** gateway to an interactive **Streamlit** dashboard.

---

## 🏗️ Architecture & Data Flow

```
   ┌────────────────────────────────────────────────────────┐
   │                   Traffic Simulator                    │
   │    (Healthy Traffic / Bad Schema / Distribution Drift) │
   └───────────────────────────┬────────────────────────────┘
                               │ JSON Events (100 events/sec)
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │             Apache Kafka (Port 9092)                   │
   │            Topic: fraud_transactions                   │
   └───────────────────────────┬────────────────────────────┘
                               │ Streaming Consumer
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │               Stream Processing Worker                 │
   │  ┌───────────────────────┐   ┌──────────────────────┐  │
   │  │    DataValidator      │   │     DriftEngine      │  │
   │  │ (Type & Bound Checks) │   │ (KS-Test & PSI Math) │  │
   │  └───────────────────────┘   └──────────────────────┘  │
   └───────────────────────────┬────────────────────────────┘
                               │ Writes Alerts & Drift Metrics
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │               PostgreSQL (Port 5433)                   │
   │     Tables: data_quality_alerts, drift_metrics         │
   └───────────────────────────┬────────────────────────────┘
                               │ Queries DB via SQLAlchemy
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │             FastAPI Backend (Port 8000)                │
   │           REST Endpoints: /api/dashboard/*             │
   └───────────────────────────┬────────────────────────────┘
                               │ Polling REST API
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │           Streamlit Dashboard (Port 8501)              │
   │        Live Feature Drift Status & Alert Feed          │
   └────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Python 3.12+ / 3.13**
- **Docker Desktop** (Running)

### 2. Clone and Install Dependencies
```bash
git clone https://github.com/your-username/driftguard.git
cd driftguard
pip install -r requirements.txt
```

### 3. Spin up Docker Infrastructure
Launch Kafka, Zookeeper, PostgreSQL, Prometheus, and Grafana:
```bash
docker compose up -d
```

### 4. Train Model & Initialize Database
```bash
# 1. Generate baseline dataset & train XGBoost model
python data/generate_dataset.py
python data/train_model.py

# 2. Create PostgreSQL tables
python stream_processor/models.py
```

### 5. Launch the Platform (Separate Terminals)

* **Terminal 1: Stream Processing Worker**
  ```bash
  python stream_processor/main.py
  ```
* **Terminal 2: FastAPI Backend**
  ```bash
  python api_server/main.py
  ```
* **Terminal 3: Streamlit UI Dashboard**
  ```bash
  python -m streamlit run dashboard/app.py --server.headless true
  ```
* **Terminal 4: Production Traffic Simulator**
  ```bash
  python simulator/simulator.py
  ```

---

## 📊 Live Observability Dashboard

Open **[http://localhost:8501](http://localhost:8501)** in your browser.

As the simulator cycles through traffic patterns, you will observe:
- **Healthy Phase (0 - 1000 events)**: Model Health shows `100%`, all features green.
- **Data Quality Anomaly Phase (1000 - 1050 events)**: The **Recent Data Quality Alerts** table logs schema violations (e.g., negative incomes, age bounds).
- **Drift Phase (2000 - 3500 events)**: Features like `user_age` and `transaction_amount` shift to **Critical (red)** with elevated KS-test scores.

---

## 📂 Project Structure

```text
├── api_server/
│   └── main.py                     # FastAPI REST API exposing metrics and alerts
├── dashboard/
│   └── app.py                      # Streamlit real-time monitoring dashboard
├── data/
│   ├── generate_dataset.py         # Synthetic transaction dataset generator
│   ├── train_model.py              # XGBoost training & statistical baseline profiling
│   ├── baseline_dataset.csv        # Baseline reference data (100k samples)
│   └── baseline_stats.json         # Reference statistics & quantiles
├── simulator/
│   └── simulator.py                # Event generator with dynamic drift & fault injection
├── stream_processor/
│   ├── main.py                     # Kafka consumer orchestrating validation & drift checks
│   ├── validator.py                # Schema limits, null checks, and type validation
│   ├── drift_engine.py             # Math engine implementing KS-Test and PSI
│   └── models.py                   # SQLAlchemy ORM models for PostgreSQL
├── docker-compose.yml              # Multi-container orchestration (Kafka, Postgres, Grafana)
├── prometheus.yml                  # Prometheus metric scraping config
├── requirements.txt                # Project dependencies
├── PROJECT_GUIDE.md                # In-depth technical guide, statistical formulas & theory
└── README.md                       # Project overview
```

---

## 📖 Deep-Dive Documentation
For a complete mathematical explanation of **KS-Test**, **Population Stability Index (PSI)**, failure scenarios, and interview talking points:
👉 **Read the [PROJECT_GUIDE.md](PROJECT_GUIDE.md)**

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
