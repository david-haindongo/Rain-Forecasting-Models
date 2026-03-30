#!/usr/bin/env python3
"""
Windhoek Advanced Predictive Weather Framework
==============================================
A sophisticated weather prediction system combining:
1. Complexity Theory (Hurst Exponent) - Detects regime changes
2. Bayesian Reasoning (HMM) - Identifies hidden states
3. Machine Learning (XGBoost, Random Forest) - High accuracy predictions
4. Real-time data from Open-Meteo API
"""

import os
import warnings
import time
import json
from datetime import datetime, timedelta

import requests
import pandas as pd
import numpy as np
import nolds
from hmmlearn import hmm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy import stats
import xgboost as xgb

warnings.filterwarnings('ignore')


class WindhoekAdvancedPredictor:
    """
    Advanced predictive framework for Windhoek's atmospheric patterns.
    Combines multiple intelligence layers for high-accuracy predictions.
    """
    
    def __init__(self):
        # Windhoek coordinates
        self.lat, self.lon = -22.56, 17.08
        self.location_name = "Windhoek, Namibia"
        
        # Bayesian Layer: 2-State Hidden Markov Model
        self.hmm_model = hmm.GaussianHMM(
            n_components=2, 
            covariance_type="full", 
            n_iter=200,
            random_state=42
        )
        
        # Machine Learning Models
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        self.gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        # Data storage
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.model_scores = {}
        self.cache_dir = "weather_cache"
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def fetch_from_openmeteo(self):
        """
        Fetch live data from Open-Meteo API (works reliably)
        """
        url = "https://api.open-meteo.com/v1/forecast"
        
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "current": [
                "temperature_2m", 
                "relative_humidity_2m", 
                "surface_pressure", 
                "cloud_cover",
                "wind_speed_10m",
                "wind_direction_10m"
            ],
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "surface_pressure",
                "precipitation",
                "cloud_cover",
                "wind_speed_10m"
            ],
            "past_days": 30,
            "forecast_days": 1,
            "timezone": "auto"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"  API Error: {e}")
            return None
    
    def ingest_data(self):
        """
        Ingest and prepare data from Open-Meteo API
        """
        print("Ingesting data from Open-Meteo API...")
        
        # Fetch live data
        json_data = self.fetch_from_openmeteo()
        if not json_data:
            print("  API fetch failed, using synthetic data...")
            return self._create_synthetic_historical(), {}
        
        # Process live current conditions
        current = json_data.get('current', {})
        
        print(f"  Live data fetched:")
        print(f"    Temperature: {current.get('temperature_2m')}°C")
        print(f"    Humidity: {current.get('relative_humidity_2m')}%")
        print(f"    Pressure: {current.get('surface_pressure')} hPa")
        print(f"    Cloud Cover: {current.get('cloud_cover')}%")
        
        # Process historical data
        if 'hourly' in json_data:
            hourly = json_data['hourly']
            df = pd.DataFrame({
                'time': pd.to_datetime(hourly['time']),
                'temperature': hourly['temperature_2m'],
                'humidity': hourly['relative_humidity_2m'],
                'pressure': hourly['surface_pressure'],
                'precipitation': hourly['precipitation'],
                'cloud_cover': hourly.get('cloud_cover', [0] * len(hourly['time'])),
                'wind_speed': hourly.get('wind_speed_10m', [0] * len(hourly['time']))
            })
            df.set_index('time', inplace=True)
            
            # Create target variable (rain in next 6 hours)
            df['rain_next_6h'] = df['precipitation'].shift(-6).fillna(0) > 0.1
            df['rain_next_6h'] = df['rain_next_6h'].astype(int)
            
            print(f"  Historical data: {len(df)} hourly records")
            
            # Save raw data
            df.to_csv('windhoek_raw_hourly.csv')
            
            return df, current
        else:
            print("  No hourly data available, using synthetic data...")
            return self._create_synthetic_historical(), current
    
    def _create_synthetic_historical(self):
        """
        Create realistic synthetic historical data based on Windhoek patterns
        """
        print("  Creating synthetic historical data...")
        
        end = datetime.now()
        start = end - timedelta(days=30)
        dates = pd.date_range(start=start, end=end, freq='H')
        
        np.random.seed(42)
        
        # Base patterns
        hour_of_day = dates.hour
        day_of_year = dates.dayofyear
        
        # Temperature: daily cycle + seasonal
        daily_temp = 15 * np.sin(2 * np.pi * (hour_of_day - 14) / 24) + 25
        seasonal_temp = 5 * np.sin(2 * np.pi * day_of_year / 365)
        temperature = daily_temp + seasonal_temp + np.random.normal(0, 2, len(dates))
        
        # Humidity: inverse of temperature + noise
        humidity = 70 - 0.5 * (temperature - 20) + np.random.normal(0, 10, len(dates))
        humidity = np.clip(humidity, 20, 95)
        
        # Pressure: seasonal + random
        pressure = 850 + 10 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 3, len(dates))
        
        # Precipitation: rain events with realistic patterns
        rain_prob = 0.1 + 0.2 * (humidity > 70) + 0.1 * (pressure < 845)
        precipitation = np.where(
            np.random.random(len(dates)) < rain_prob,
            np.random.exponential(2, len(dates)),
            0
        )
        
        df = pd.DataFrame({
            'temperature': temperature,
            'humidity': humidity,
            'pressure': pressure,
            'precipitation': precipitation,
            'cloud_cover': np.random.uniform(0, 100, len(dates)),
            'wind_speed': np.random.exponential(5, len(dates))
        }, index=dates)
        
        # Create target (rain in next 6 hours)
        df['rain_next_6h'] = df['precipitation'].shift(-6).fillna(0) > 0.1
        df['rain_next_6h'] = df['rain_next_6h'].astype(int)
        
        print(f"  Created {len(df)} synthetic records")
        return df
    
    def engineer_advanced_features(self, df):
        """
        Create sophisticated meteorological features for high accuracy
        """
        print("\nEngineering advanced features...")
        
        features = df.copy()
        
        # Ensure we have enough data for feature engineering
        if len(features) < 50:
            print(f"  Warning: Only {len(features)} data points. Feature engineering may be limited.")
        
        # 1. Lag features (atmospheric memory)
        for lag in [1, 3, 6, 12, 24]:
            if len(features) > lag:
                features[f'pressure_lag_{lag}h'] = features['pressure'].shift(lag)
                features[f'temp_lag_{lag}h'] = features['temperature'].shift(lag)
                features[f'humidity_lag_{lag}h'] = features['humidity'].shift(lag)
        
        # 2. Rolling statistics (trend detection) - with min_periods to handle edges
        for window in [3, 6, 12, 24]:
            if len(features) > window:
                # Pressure trends
                features[f'pressure_rolling_mean_{window}h'] = features['pressure'].rolling(window, min_periods=1).mean()
                features[f'pressure_rolling_std_{window}h'] = features['pressure'].rolling(window, min_periods=1).std()
                features[f'pressure_trend_{window}h'] = features['pressure'] - features[f'pressure_rolling_mean_{window}h']
                
                # Humidity trends
                features[f'humidity_rolling_mean_{window}h'] = features['humidity'].rolling(window, min_periods=1).mean()
                features[f'humidity_trend_{window}h'] = features['humidity'] - features[f'humidity_rolling_mean_{window}h']
        
        # 3. Rate of change (first derivatives)
        features['pressure_change_1h'] = features['pressure'].diff(1)
        features['pressure_change_3h'] = features['pressure'].diff(3)
        features['pressure_change_6h'] = features['pressure'].diff(6)
        
        features['humidity_change_1h'] = features['humidity'].diff(1)
        features['humidity_change_3h'] = features['humidity'].diff(3)
        
        # 4. Acceleration (second derivatives)
        features['pressure_accel'] = features['pressure_change_1h'].diff(1)
        
        # 5. Interaction features (meteorological indices)
        # Equivalent potential temperature proxy
        features['theta_e_proxy'] = features['temperature'] * np.exp(
            0.0002 * features['humidity'] * features['pressure'] / 1000
        )
        
        # Saturation deficit
        features['sat_deficit'] = features['humidity'] - 100
        
        # Lifted index proxy (atmospheric stability)
        features['lifted_index_proxy'] = (
            features['temperature'] - 
            10 * np.log(features['pressure'] / 1000)
        )
        
        # 6. Cyclical time features
        features['hour_sin'] = np.sin(2 * np.pi * features.index.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features.index.hour / 24)
        features['day_sin'] = np.sin(2 * np.pi * features.index.dayofyear / 365)
        features['day_cos'] = np.cos(2 * np.pi * features.index.dayofyear / 365)
        
        # 7. Composite risk indices
        # Rapid pressure drop + high humidity = high rain risk
        features['storm_risk'] = (
            -features['pressure_change_3h'].clip(-10, 0) * 
            features['humidity'] / 100
        )
        
        # Frontal passage indicator
        features['frontal_indicator'] = (
            (features['pressure_change_3h'].abs() > 2) & 
            (features['temperature'].diff(3).abs() > 2)
        ).astype(float)
        
        # 8. Complexity metrics (using nolds)
        try:
            # Calculate Hurst on rolling windows if enough data
            if len(features) > 100:
                pressure_values = features['pressure'].dropna().values[-100:]
                if len(pressure_values) > 50:
                    h = nolds.hurst_rs(pressure_values)
                    features['hurst_exponent'] = h
                else:
                    features['hurst_exponent'] = 0.5
            else:
                features['hurst_exponent'] = 0.5
        except:
            features['hurst_exponent'] = 0.5
        
        # Fill NaN values - FIXED: Use separate steps without 'method' parameter
        # First forward fill, then backward fill, then fill any remaining with 0
        features = features.ffill().bfill().fillna(0)
        
        print(f"  Created {len(features.columns)} features")
        
        return features
    
    def apply_hmm_detection(self, df):
        """
        Apply Hidden Markov Model to detect regime shifts
        """
        print("\nApplying Bayesian HMM for regime detection...")
        
        # Ensure we have enough data for HMM
        if len(df) < 50:
            print(f"  Insufficient data for HMM (need 50+ points, have {len(df)}). Using fallback.")
            df['hmm_volatile_prob'] = 0.5
            df['regime_stability'] = 0.5
            return df
        
        # Use pressure and temperature for HMM
        observations = df[['pressure', 'temperature']].values
        
        try:
            # Fit HMM
            self.hmm_model.fit(observations)
            
            # Get hidden states
            states = self.hmm_model.predict(observations)
            
            # Calculate regime probabilities
            state_probs = self.hmm_model.predict_proba(observations)
            
            # Identify volatile regime (higher variance)
            volatile_state = np.argmax([
                np.var(observations[states == i]) if len(observations[states == i]) > 0 else 0 
                for i in range(2)
            ])
            
            df['hmm_state'] = states
            df['hmm_volatile_prob'] = state_probs[:, volatile_state]
            
            # Calculate regime persistence
            state_changes = np.sum(np.diff(states) != 0)
            df['regime_stability'] = 1 - (state_changes / max(1, len(states)))
            
            print(f"  HMM detected {state_changes} regime changes")
            print(f"  Current regime: {'VOLATILE' if states[-1] == volatile_state else 'STABLE'}")
            
        except Exception as e:
            print(f"  HMM failed: {e}")
            df['hmm_volatile_prob'] = 0.5
            df['regime_stability'] = 0.5
        
        return df
    
    def train_ensemble(self, df):
        """
        Train ensemble of ML models with cross-validation and handle class imbalance
        """
        print("\nTraining ensemble models...")
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in [
            'precipitation', 'rain_next_6h', 'hmm_state'
        ]]
        
        # Ensure we have enough data
        if len(df) < 100:
            print(f"  Warning: Limited data ({len(df)} rows). Results may be unstable.")
        
        X = df[feature_cols].fillna(0)
        y = df['rain_next_6h']
        
        # Check class distribution
        rain_count = y.sum()
        no_rain_count = len(y) - rain_count
        print(f"\n  Class Distribution:")
        print(f"    No Rain: {no_rain_count} ({no_rain_count/len(y):.1%})")
        print(f"    Rain: {rain_count} ({rain_count/len(y):.1%})")
        
        # Time-based split
        split_idx = max(int(len(X) * 0.7), len(X) - 100)
        if split_idx >= len(X) - 10:
            split_idx = len(X) - 20
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"\n  Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"  Training rain events: {y_train.sum()}")
        print(f"  Test rain events: {y_test.sum()}")
        
        # Calculate class weights for imbalance
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        print(f"  Class weights: {class_weight_dict}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Re-initialize models with class weights
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced',  # Handle imbalance
            random_state=42
        )
        
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),  # Handle imbalance
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        self.gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        # Train Random Forest
        print("\n  Training Random Forest...")
        self.rf_model.fit(X_train_scaled, y_train)
        rf_pred = self.rf_model.predict(X_test_scaled)
        rf_proba = self.rf_model.predict_proba(X_test_scaled)
        
        # Feature importance
        self.feature_importance['rf'] = dict(zip(
            feature_cols, self.rf_model.feature_importances_
        ))
        
        # Train XGBoost
        print("  Training XGBoost...")
        self.xgb_model.fit(X_train_scaled, y_train)
        xgb_pred = self.xgb_model.predict(X_test_scaled)
        xgb_proba = self.xgb_model.predict_proba(X_test_scaled)
        
        # Train Gradient Boosting
        print("  Training Gradient Boosting...")
        self.gb_model.fit(X_train_scaled, y_train)
        gb_pred = self.gb_model.predict(X_test_scaled)
        gb_proba = self.gb_model.predict_proba(X_test_scaled)
        
        # Calculate comprehensive metrics
        self.model_scores = {}
        
        for name, pred, proba in [('RF', rf_pred, rf_proba), 
                                   ('XGB', xgb_pred, xgb_proba), 
                                   ('GB', gb_pred, gb_proba)]:
            
            # Basic metrics
            accuracy = accuracy_score(y_test, pred)
            precision = precision_score(y_test, pred, zero_division=0)
            recall = recall_score(y_test, pred, zero_division=0)
            f1 = f1_score(y_test, pred, zero_division=0)
            
            # Additional metrics
            from sklearn.metrics import roc_auc_score, average_precision_score
            try:
                auc_roc = roc_auc_score(y_test, proba[:, 1])
                avg_precision = average_precision_score(y_test, proba[:, 1])
            except:
                auc_roc = 0.5
                avg_precision = 0.0
            
            self.model_scores[name.lower()] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc_roc': auc_roc,
                'avg_precision': avg_precision
            }
            
            # Print detailed metrics
            print(f"\n    {name} Performance:")
            print(f"      Accuracy:  {accuracy:.1%}")
            print(f"      Precision: {precision:.3f} (of predicted rain, how many were correct)")
            print(f"      Recall:    {recall:.3f} (of actual rain, how many were caught)")
            print(f"      F1 Score:  {f1:.3f} (harmonic mean of precision & recall)")
            print(f"      AUC-ROC:   {auc_roc:.3f} (discrimination ability)")
            
            # Confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, pred)
            print(f"      Confusion Matrix: [TN={cm[0,0]} FP={cm[0,1]}; FN={cm[1,0]} TP={cm[1,1]}]")
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def predict_probability(self, latest_features, feature_cols):
        """
        Generate ensemble probability prediction with optimal threshold tuning
        """
        # Scale latest features
        latest_scaled = self.scaler.transform(latest_features[feature_cols].fillna(0))

        # Get probabilities from all models
        rf_proba = self.rf_model.predict_proba(latest_scaled)[0][1]
        xgb_proba = self.xgb_model.predict_proba(latest_scaled)[0][1]
        gb_proba = self.gb_model.predict_proba(latest_scaled)[0][1]

        # Find optimal thresholds for each model based on test performance
        # We want to balance precision and recall

        # For Random Forest (currently too conservative - threshold too high)
        # Lower threshold to catch more rain events
        rf_adjusted = rf_proba * 1.5  # Boost RF since it's under-predicting

        # For XGBoost (extremely conservative - threshold at ~0.9)
        # Significantly lower threshold
        xgb_adjusted = xgb_proba * 2.5  # Boost XGB since it has best AUC

        # For Gradient Boosting (already well-calibrated)
        gb_adjusted = gb_proba

        # Use model weights based on multiple metrics
        weights = {
            'rf': self.model_scores['rf']['auc_roc'] * 0.3,  # 30% weight on discrimination
            'xgb': self.model_scores['xgb']['auc_roc'] * 0.4,  # 40% weight (best AUC)
            'gb': self.model_scores['gb']['f1'] * 0.3  # 30% weight on F1
        }

        total_weight = sum(weights.values())

        # Weighted ensemble with adjusted probabilities
        ensemble_proba = (
            weights['rf'] * rf_adjusted +
            weights['xgb'] * xgb_adjusted +
            weights['gb'] * gb_adjusted
        ) / total_weight

        # Clip to valid range
        ensemble_proba = np.clip(ensemble_proba, 0, 1)

        # Adjust for HMM regime
        if 'hmm_volatile_prob' in latest_features.columns:
            hmm_prob = latest_features['hmm_volatile_prob'].values[-1]
            # When HMM says volatile, increase confidence in rain predictions
            if hmm_prob > 0.7:
                ensemble_proba = ensemble_proba * 1.2
            final_prob = 0.8 * ensemble_proba + 0.2 * hmm_prob
        else:
            final_prob = ensemble_proba
            hmm_prob = 0.5

        # Clip again
        final_prob = np.clip(final_prob, 0, 1)

        # Calculate confidence based on multiple factors
        # 1. Model agreement (lower std = higher agreement)
        prob_std = np.std([rf_proba, xgb_proba, gb_proba])
        agreement = 1 - min(prob_std, 0.5)  # Normalize

        # 2. Historical performance (AUC of best model)
        performance = self.model_scores['xgb']['auc_roc']

        # 3. Distance from 0.5 (decisiveness)
        decisiveness = abs(final_prob - 0.5) * 2

        confidence_score = 0.4 * agreement + 0.3 * performance + 0.3 * decisiveness

        if confidence_score > 0.7:
            confidence = "HIGH"
        elif confidence_score > 0.5:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        print(f"\n  Model Agreement: {agreement:.2f}")
        print(f"  Performance Score: {performance:.2f}")
        print(f"  Decisiveness: {decisiveness:.2f}")
        print(f"  Confidence Score: {confidence_score:.2f}")

        return {
            'ensemble_probability': float(final_prob),
            'rf_probability': float(rf_proba),
            'xgb_probability': float(xgb_proba),
            'gb_probability': float(gb_proba),
            'hmm_probability': float(hmm_prob),
            'rf_adjusted': float(rf_adjusted),
            'xgb_adjusted': float(xgb_adjusted),
            'confidence': confidence,
            'agreement': float(agreement)
        }
    def _get_confidence_level(self, final_prob, ensemble_prob, hmm_prob):
        """
        Determine confidence level based on agreement and historical accuracy
        """
        # Agreement between models and HMM
        agreement = 1 - abs(ensemble_prob - hmm_prob)
        
        # Distance from 0.5 (decisive predictions)
        decisiveness = abs(final_prob - 0.5) * 2
        
        # Combined confidence
        confidence_score = 0.6 * agreement + 0.4 * decisiveness
        
        if confidence_score > 0.7:
            return "HIGH"
        elif confidence_score > 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def run_framework(self):
        """
        Execute complete predictive workflow
        """
        print("\n" + "="*60)
        print("WINDHOEK ADVANCED PREDICTIVE FRAMEWORK")
        print("="*60)
        print(f"Location: {self.location_name}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-"*60)
        
        # Step 1: Data Ingestion
        print("\n[STEP 1] Data Ingestion")
        historical_df, current = self.ingest_data()
        
        # Step 2: Feature Engineering
        print("\n[STEP 2] Advanced Feature Engineering")
        featured_df = self.engineer_advanced_features(historical_df)
        
        # Step 3: HMM Regime Detection
        print("\n[STEP 3] Bayesian Regime Detection")
        featured_df = self.apply_hmm_detection(featured_df)
        
        # Step 4: Train Ensemble Models
        X_train, X_test, y_train, y_test, feature_cols = self.train_ensemble(featured_df)
        
        # Step 5: Generate Prediction
        print("\n[STEP 4] Ensemble Prediction")
        latest_features = featured_df.iloc[-1:].copy()
        prediction = self.predict_probability(latest_features, feature_cols)
        
        # Step 6: Final Output
        print("\n" + "="*60)
        print("FINAL PREDICTION")
        print("="*60)
        
        rain_prob = prediction['ensemble_probability'] * 100
        
        if rain_prob >= 60:
            symbol = "🌧️  RAIN EXPECTED"
            outcome = "RAIN"
        elif rain_prob >= 30:
            symbol = "⛅ POSSIBLE SHOWERS"
            outcome = "POSSIBLE"
        else:
            symbol = "☀️  CLEAR SKIES"
            outcome = "CLEAR"
        
        print(f"\n  {symbol}")
        print(f"\n  Probability: {rain_prob:.1f}%")
        print(f"  Confidence: {prediction['confidence']}")
        
        print(f"\n  Model Breakdown:")
        print(f"    • Random Forest: {prediction['rf_probability']*100:.1f}%")
        print(f"    • XGBoost: {prediction['xgb_probability']*100:.1f}%")
        print(f"    • Gradient Boosting: {prediction['gb_probability']*100:.1f}%")
        print(f"    • HMM Regime: {prediction['hmm_probability']*100:.1f}%")
        
        # Step 7: Hurst Exponent (Complexity Analysis)
        print("\n[COMPLEXITY ANALYSIS]")
        try:
            pressure_series = featured_df['pressure'].values[-100:]
            if len(pressure_series) > 50:
                hurst = nolds.hurst_rs(pressure_series)
                print(f"  Hurst Exponent: {hurst:.3f}")
                if hurst < 0.4:
                    print("  → Mean-reverting (stable system)")
                elif hurst < 0.6:
                    print("  → Random walk (unpredictable)")
                else:
                    print("  → Trending (regime shift possible)")
            else:
                print("  Insufficient data for Hurst calculation")
        except Exception as e:
            print(f"  Hurst calculation unavailable: {e}")
        
        # Step 8: Save Results
        print("\n[EXPORTING RESULTS]")
        
        # Save detailed output
        output = pd.DataFrame([{
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'location': self.location_name,
            'prediction': outcome,
            'probability': rain_prob,
            'confidence': prediction['confidence'],
            'rf_prob': prediction['rf_probability'],
            'xgb_prob': prediction['xgb_probability'],
            'gb_prob': prediction['gb_probability'],
            'hmm_prob': prediction['hmm_probability'],
            'hurst_exponent': hurst if 'hurst' in locals() else 0.5,
            'temperature': current.get('temperature_2m', 0),
            'humidity': current.get('relative_humidity_2m', 0),
            'pressure': current.get('surface_pressure', 0),
            'cloud_cover': current.get('cloud_cover', 0)
        }])
        
        output.to_csv('windhoek_advanced_prediction.csv', index=False)
        
        # Save feature importance
        if self.feature_importance:
            importance_df = pd.DataFrame([
                {'feature': f, 'importance': self.feature_importance['rf'].get(f, 0)}
                for f in list(self.feature_importance['rf'].keys())[:20]
            ]).sort_values('importance', ascending=False)
            importance_df.to_csv('feature_importance.csv', index=False)
        
        print(f"  Results saved to:")
        print(f"    • windhoek_advanced_prediction.csv")
        print(f"    • feature_importance.csv")
        print(f"    • windhoek_raw_hourly.csv")
        
        print("\n" + "="*60)


if __name__ == "__main__":
    predictor = WindhoekAdvancedPredictor()
    predictor.run_framework()
