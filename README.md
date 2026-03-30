# Rain Forecasting Models (Windhoek, Namibia)

This repository contains an **ensemble forecasting system** designed to predict rainfall patterns in the Windhoek region. The project leverages multiple machine learning architectures to provide robust, high-accuracy weather predictions as part of a Master of Data Science research initiative.

## 🚀 System Overview
The core of this project is the **Windhoek Rain Predictor (v14.0)**, which utilizes an ensemble approach to mitigate the variance of individual models.

* **Ensemble Architecture**: Combines eight distinct base models to generate a consensus forecast.
* **Data Sources**: Integrates historical weather data and real-time airport observations (e.g., `airports.csv`).
* **Stack**: Built with Python, Flask for the dashboard, and specialized libraries including CatBoost and XGBoost.

## 📊 Key Features
* **Predictive Modeling**: High-fidelity forecasting specifically tuned for the unique climate of central Namibia.
* **Interactive Dashboard**: A Flask-based web interface for visualizing model outputs and historical trends.
* **Forensic Analysis**: Comprehensive logging (e.g., `logs_rain_predictor_v14.log`) for model performance auditing.

## 🛠️ Setup & Installation
To run the predictor locally without the large environment files previously excluded, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/david-haindongo/Rain-Forecasting-Models.git
   cd Rain-Forecasting-Models
   ```

2. **Rebuild the environment:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Dashboard:**
   ```bash
   python app.py
   ```

## 📂 Project Structure
* `david_rain_predictor14.py`: The primary ensemble logic and model definitions.
* `app.py`: Flask application server.
* `templates/`: UI components for the web dashboard.
* `weather_cache/`: Local data storage for accelerated inference.
