# ============================================================================
# HFT QUANTITATIVE TRADING SYSTEM - INSTITUTIONAL DATASET
# ============================================================================

# INSTALLATION REQUIREMENTS - Run this cell first in Google Colab
"""
!pip install numpy pandas openpyxl scikit-learn statsmodels arch scipy matplotlib seaborn tensorflow tqdm
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from arch import arch_model
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
import collections
import random

from tqdm import tqdm
import time

# ============================================================================
# SECTION 1: ADVANCED DATA LOADING FOR INSTITUTIONAL DATASET
# ============================================================================

def load_institutional_data(filepath):
    """Load and prepare institutional-grade dataset with 230 features"""
    df = pd.read_excel(filepath, sheet_name='Enhanced_Daily_Per_Symbol')
    
    print(f"Raw data shape: {df.shape}")
    print(f"Columns detected: {len(df.columns)}")
    
    # Ensure proper datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        df['timestamp'] = df['Date']
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Handle extreme values and errors
    for col in numeric_cols:
        # Replace infinities
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # Cap extreme values (> 1e10 or < -1e10)
        mask_high = df[col] > 1e10
        mask_low = df[col] < -1e10
        if mask_high.any():
            print(f"  Capping extreme high values in {col}")
            df.loc[mask_high, col] = df.loc[~mask_high, col].quantile(0.99)
        if mask_low.any():
            print(f"  Capping extreme low values in {col}")
            df.loc[mask_low, col] = df.loc[~mask_low, col].quantile(0.01)
    
    # Core price data
    price_col = 'Close' if 'Close' in df.columns else 'Adj Close'
    df['price'] = df[price_col].fillna(method='ffill').fillna(method='bfill')
    df['volume'] = df['Volume'].fillna(method='ffill').fillna(method='bfill') if 'Volume' in df.columns else 100000
    
    # Returns
    if 'Daily_Return' not in df.columns:
        df['Daily_Return'] = df['price'].pct_change()
    
    df['returns'] = df['Daily_Return'].fillna(0)
    
    # Forward fill then backward fill all numeric columns
    print("Cleaning missing values...")
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            # If still NaN, fill with column median
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
            # If still NaN (all values were NaN), fill with 0
            if df[col].isnull().any():
                df[col] = df[col].fillna(0)
    
    # Drop rows where price or returns are still missing (critical columns)
    df = df.dropna(subset=['price'])
    df = df.reset_index(drop=True)
    
    print(f"Cleaned data shape: {df.shape}")
    print(f"Remaining NaN values: {df.isnull().sum().sum()}")
    
    return df

# ============================================================================
# SECTION 2: INTELLIGENT FEATURE SELECTION
# ============================================================================

def select_intelligent_features(df):
    """Select best features from 230-column institutional dataset"""
    
    # Core features (always include)
    core_features = ['price', 'volume', 'returns', 'Daily_Return']
    
    # Technical indicators
    tech_features = [c for c in df.columns if any(x in c for x in 
                    ['SMA', 'EMA', 'RSI', 'MACD', 'BB_', 'Volatility'])]
    
    # Algorithmic signals
    algo_features = [c for c in df.columns if any(x in c for x in 
                    ['AMSF', 'FTCS', 'RABE', 'Hurst', 'CASP', 'ARFS', 'MHRP'])]
    
    # ML/AI features
    ml_features = [c for c in df.columns if any(x in c for x in 
                  ['AHF_', 'CAAE_', 'DFAM_', 'GNN', 'Causal', 'Federated'])]
    
    # Risk metrics
    risk_features = [c for c in df.columns if any(x in c for x in 
                    ['MRS_', 'DIR_', 'NEAD_', 'VaR', 'Drawdown', 'Uncertainty'])]
    
    # Execution metrics
    exec_features = [c for c in df.columns if any(x in c for x in 
                    ['LACM_', 'SPRINT_', 'TPS_', 'Cost', 'Spread'])]
    
    # Position sizing
    position_features = [c for c in df.columns if any(x in c for x in 
                        ['Position', 'DPS_', 'Dynamic_'])]
    
    # Final signals
    signal_features = [c for c in df.columns if any(x in c for x in 
                      ['FinalScore', 'FinalSignal', 'EnhancedFinalScore'])]
    
    # Combine all
    selected = list(set(core_features + tech_features + algo_features + 
                       ml_features + risk_features + exec_features + 
                       position_features + signal_features))
    
    # Filter to existing columns
    selected = [c for c in selected if c in df.columns]
    
    print(f"\nFeature Selection:")
    print(f"  Technical Indicators: {len(tech_features)}")
    print(f"  Algorithmic Signals: {len(algo_features)}")
    print(f"  ML/AI Features: {len(ml_features)}")
    print(f"  Risk Metrics: {len(risk_features)}")
    print(f"  Execution Metrics: {len(exec_features)}")
    print(f"  Total Selected: {len(selected)}")
    
    return selected

# ============================================================================
# SECTION 3: STATISTICAL & TIME-SERIES MODELS
# ============================================================================

class StatisticalModels:
    """Statistical arbitrage, mean reversion, ARIMA, GARCH, Kalman"""
    
    def __init__(self):
        self.models = {}
        
    def mean_reversion_zscore(self, prices, window=20):
        """Z-score based mean reversion signal"""
        ma = prices.rolling(window, min_periods=5).mean()
        std = prices.rolling(window, min_periods=5).std()
        zscore = (prices - ma) / (std + 1e-8)
        signal = -np.tanh(zscore)
        return signal.fillna(0).values
    
    def statistical_arbitrage_spread(self, price1, price2):
        """Cross-instrument spread signal"""
        spread = price1 - price2
        zscore = (spread - spread.rolling(50, min_periods=10).mean()) / \
                 (spread.rolling(50, min_periods=10).std() + 1e-8)
        signal = -np.tanh(zscore)
        return signal.fillna(0).values
    
    def fit_arima(self, returns, order=(2,0,2)):
        """ARIMA model for return forecasting with rolling window"""
        try:
            returns_clean = returns.dropna()
            if len(returns_clean) < 100:
                return np.zeros(len(returns))
            
            signal = np.zeros(len(returns))
            window = 200
            
            # Rolling ARIMA with progress bar
            total_steps = len(returns_clean) - window
            pbar = tqdm(range(window, len(returns_clean)), 
                       desc="    ARIMA rolling forecast", 
                       leave=False, 
                       ncols=100)
            
            for i in pbar:
                try:
                    train_data = returns_clean.iloc[i-window:i]
                    model = ARIMA(train_data, order=order)
                    fitted = model.fit()
                    forecast = fitted.forecast(steps=1)[0]
                    signal[i] = np.tanh(forecast * 100)
                except:
                    signal[i] = signal[i-1] if i > 0 else 0
            
            return signal
        except Exception as e:
            print(f"ARIMA warning: {str(e)[:50]}")
            return np.zeros(len(returns))
    
    def fit_garch(self, returns):
        """GARCH model for volatility forecasting with rolling window"""
        try:
            returns_clean = returns.dropna() * 100
            if len(returns_clean) < 100:
                return np.zeros(len(returns))
            
            signal = np.zeros(len(returns))
            window = 200
            
            # Rolling GARCH with progress bar
            pbar = tqdm(range(window, len(returns_clean)), 
                       desc="    GARCH rolling forecast", 
                       leave=False, 
                       ncols=100)
            
            for i in pbar:
                try:
                    train_data = returns_clean.iloc[i-window:i]
                    model = arch_model(train_data, vol='Garch', p=1, q=1)
                    fitted = model.fit(disp='off', show_warning=False)
                    forecast = fitted.forecast(horizon=1)
                    vol_forecast = np.sqrt(forecast.variance.values[-1, 0])
                    
                    ret_std = train_data.std()
                    if ret_std > 0:
                        signal[i] = -np.tanh((vol_forecast - ret_std) / ret_std)
                except:
                    signal[i] = signal[i-1] if i > 0 else 0
            
            return signal
        except Exception as e:
            print(f"GARCH warning: {str(e)[:50]}")
            return np.zeros(len(returns))
    
    def kalman_filter(self, prices):
        """Kalman filter for hidden state estimation"""
        n = len(prices)
        if n == 0:
            return np.array([])
        
        x = np.zeros(n)
        P = np.zeros(n)
        
        prices_array = prices.values if hasattr(prices, 'values') else np.array(prices)
        x[0] = prices_array[0]
        P[0] = 1.0
        
        Q = 0.001
        R = 0.1
        
        for t in range(1, n):
            x_pred = x[t-1]
            P_pred = P[t-1] + Q
            K = P_pred / (P_pred + R)
            x[t] = x_pred + K * (prices_array[t] - x_pred)
            P[t] = (1 - K) * P_pred
        
        price_std = np.std(prices_array)
        if price_std == 0:
            price_std = 1.0
        signal = np.tanh((prices_array - x) / price_std)
        return signal
    
    def fit_all(self, df):
        """Fit all statistical models"""
        signals = pd.DataFrame(index=df.index)
        
        print("  - Mean Reversion (Z-Score)...")
        signals['mean_rev'] = self.mean_reversion_zscore(df['price'])
        
        print("  - Statistical Arbitrage...")
        if 'SMA_50' in df.columns:
            signals['stat_arb'] = self.statistical_arbitrage_spread(
                df['price'], df['SMA_50'])
        else:
            signals['stat_arb'] = self.statistical_arbitrage_spread(
                df['price'], df['price'].shift(1))
        
        print("  - ARIMA...")
        signals['arima'] = self.fit_arima(df['returns'])
        
        print("  - GARCH...")
        signals['garch'] = self.fit_garch(df['returns'])
        
        print("  - Kalman Filter...")
        signals['kalman'] = self.kalman_filter(df['price'])
        
        return signals.fillna(0)

# ============================================================================
# SECTION 4: MACHINE LEARNING MODELS
# ============================================================================

class MLModels:
    """SVM, Random Forest leveraging institutional features"""
    
    def __init__(self):
        self.models = {}
        self.scaler = RobustScaler()
        self.imputer = None
        
    def prepare_ml_features(self, df, feature_cols):
        """Extract and prepare features for ML with comprehensive cleaning"""
        from sklearn.impute import SimpleImputer
        
        # Extract features
        X = df[feature_cols].copy()
        
        # Additional cleaning
        print(f"    Feature matrix shape: {X.shape}")
        print(f"    NaN count before cleaning: {X.isnull().sum().sum()}")
        
        # Replace inf/-inf with NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Impute missing values using median strategy
        if self.imputer is None:
            self.imputer = SimpleImputer(strategy='median')
            X_clean = self.imputer.fit_transform(X)
        else:
            X_clean = self.imputer.transform(X)
        
        print(f"    NaN count after imputation: {np.isnan(X_clean).sum()}")
        
        # Final check - replace any remaining NaN with 0
        X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create target
        y_reg = df['returns'].shift(-1).fillna(0).values
        y_clf = (y_reg > 0).astype(int)
        
        print(f"    Final feature matrix: {X_clean.shape}, clean: {not np.isnan(X_clean).any()}")
        
        return X_clean, y_reg, y_clf
    
    def train_test_split_ts(self, X, y, test_size=0.2):
        """Time-series train-test split"""
        split_idx = int(len(X) * (1 - test_size))
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
    
    def fit_svm(self, X_train, y_train, X_test):
        """Support Vector Machine with robust preprocessing"""
        # Scale data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Additional NaN check
        if np.isnan(X_train_scaled).any():
            print("    WARNING: NaN detected after scaling, applying final cleanup")
            X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)
        if np.isnan(X_test_scaled).any():
            X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0)
        
        # Use subset for speed
        train_size = min(5000, len(X_train))
        idx = np.random.choice(len(X_train), train_size, replace=False)
        
        model = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale')
        model.fit(X_train_scaled[idx], y_train[idx])
        
        pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        signal = (pred_proba - 0.5) * 2
        
        self.models['svm'] = model
        return signal
    
    def fit_random_forest(self, X_train, y_train, X_test):
        """Random Forest with NaN handling"""
        # Final NaN check
        if np.isnan(X_train).any():
            print("    WARNING: NaN in Random Forest input, cleaning...")
            X_train = np.nan_to_num(X_train, nan=0.0)
        if np.isnan(X_test).any():
            X_test = np.nan_to_num(X_test, nan=0.0)
        
        model = RandomForestClassifier(
            n_estimators=100, max_depth=15, min_samples_split=20,
            n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
        
        pred_proba = model.predict_proba(X_test)[:, 1]
        signal = (pred_proba - 0.5) * 2
        
        self.models['rf'] = model
        return signal
    
    def fit_all(self, df, feature_cols):
        """Train all ML models with complete preprocessing"""
        print("  - Preparing features with imputation...")
        X, y_reg, y_clf = self.prepare_ml_features(df, feature_cols)
        X_train, X_test, y_train, y_test = self.train_test_split_ts(X, y_clf)
        
        signals = pd.DataFrame(index=df.index)
        signals['svm'] = 0.0
        signals['rf'] = 0.0
        
        print("  - Training SVM...")
        svm_signal = self.fit_svm(X_train, y_train, X_test)
        signals['svm'].iloc[-len(svm_signal):] = svm_signal
        
        print("  - Training Random Forest...")
        rf_signal = self.fit_random_forest(X_train, y_train, X_test)
        signals['rf'].iloc[-len(rf_signal):] = rf_signal
        
        return signals, X_train, X_test, y_train, y_test

# ============================================================================
# SECTION 5: DEEP LEARNING MODELS
# ============================================================================

class DeepLearningModels:
    """MLP and LSTM for sequential prediction"""
    
    def __init__(self):
        self.models = {}
        
    def build_mlp(self, input_dim):
        """Multi-layer perceptron"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_dim=input_dim),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='tanh')
        ])
        model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
        return model
    
    def build_lstm(self, seq_len, n_features):
        """LSTM for sequential prediction"""
        model = keras.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=(seq_len, n_features)),
            layers.Dropout(0.3),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='tanh')
        ])
        model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
        return model
    
    def prepare_sequences(self, X, y, seq_len=20):
        """Create sequences for LSTM"""
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i:i+seq_len])
            ys.append(y[i+seq_len])
        return np.array(Xs), np.array(ys)
    
    def fit_all(self, X_train, y_train, X_test, y_test):
        """Train all DL models with NaN handling"""
        signals = {}
        
        # Clean inputs
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_test = np.nan_to_num(y_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        print("  - Training MLP...")
        mlp = self.build_mlp(X_train.shape[1])
        mlp.fit(X_train, y_train, epochs=20, batch_size=64, 
                verbose=0, validation_split=0.1)
        mlp_pred = mlp.predict(X_test, verbose=0).flatten()
        signals['mlp'] = mlp_pred
        self.models['mlp'] = mlp
        
        print("  - Training LSTM...")
        seq_len = 20
        X_train_seq, y_train_seq = self.prepare_sequences(X_train, y_train, seq_len)
        X_test_seq, y_test_seq = self.prepare_sequences(X_test, y_test, seq_len)
        
        lstm = self.build_lstm(seq_len, X_train.shape[1])
        lstm.fit(X_train_seq, y_train_seq, epochs=20, batch_size=64,
                verbose=0, validation_split=0.1)
        lstm_pred = lstm.predict(X_test_seq, verbose=0).flatten()
        
        signals['lstm'] = np.zeros(len(X_test))
        signals['lstm'][-len(lstm_pred):] = lstm_pred
        self.models['lstm'] = lstm
        
        return signals

# ============================================================================
# SECTION 6: UNSUPERVISED LEARNING
# ============================================================================

class UnsupervisedModels:
    """Clustering and anomaly detection"""
    
    def __init__(self):
        self.models = {}
        
    def market_regime_clustering(self, X, n_clusters=4):
        """Identify market regimes"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        regimes = kmeans.fit_predict(X)
        
        # Convert to signal (-1 to 1)
        regime_signal = (regimes - regimes.mean()) / (regimes.std() + 1e-8)
        regime_signal = np.tanh(regime_signal)
        
        self.models['kmeans'] = kmeans
        return regime_signal
    
    def anomaly_detection(self, X):
        """Detect anomalous market behavior"""
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        anomalies = iso_forest.fit_predict(X)
        anomaly_scores = iso_forest.score_samples(X)
        
        # Normalize to [-1, 1]
        anomaly_signal = (anomaly_scores - anomaly_scores.mean()) / \
                         (anomaly_scores.std() + 1e-8)
        anomaly_signal = np.tanh(anomaly_signal)
        
        self.models['iso_forest'] = iso_forest
        return anomaly_signal
    
    def fit_all(self, X):
        """Fit all unsupervised models with NaN handling"""
        # Clean input
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        print("  - Market Regime Clustering...")
        regime_signal = self.market_regime_clustering(X_clean)
        
        print("  - Anomaly Detection...")
        anomaly_signal = self.anomaly_detection(X_clean)
        
        return {'regime': regime_signal, 'anomaly': anomaly_signal}

# ============================================================================
# SECTION 7: REINFORCEMENT LEARNING AGENT
# ============================================================================

class DQNAgent:
    """Deep Q-Network for trading with target network"""
    
    def __init__(self, state_size, action_size=3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = collections.deque(maxlen=5000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target()
        
    def _build_model(self):
        """Build DQN"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_dim=self.state_size),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=Adam(0.001), loss='mse')
        return model
    
    def update_target(self):
        """Update target network weights"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([x[0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])
        
        targets = self.model.predict(states, verbose=0)
        next_q = self.target_model.predict(next_states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])
        
        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train_agent(self, X_train, returns_train, episodes=10):
        """Train DQN on training data only"""
        X_clean = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        returns_clean = np.nan_to_num(returns_train, nan=0.0, posinf=0.0, neginf=0.0)
        
        position = 0
        n_samples = len(X_clean)
        
        print(f"    Training DQN for {episodes} episodes...")
        
        for ep in tqdm(range(episodes), desc="    RL Episodes", ncols=100):
            episode_reward = 0
            position = 0
            
            for t in range(n_samples - 1):
                state = X_clean[t]
                action = self.act(state, training=True)
                
                new_position = action - 1
                price_change = returns_clean[t+1]
                pnl = position * price_change
                transaction_cost = 0.0002 * abs(new_position - position)
                reward = pnl - transaction_cost
                
                episode_reward += reward
                position = new_position
                next_state = X_clean[t+1]
                done = (t == n_samples - 2)
                
                self.remember(state, action, reward, next_state, done)
                
                if len(self.memory) > 128:
                    self.replay(64)
                
                if t % 100 == 0:
                    self.update_target()
            
        return self
    
    def predict_signals(self, X):
        """Generate signals for test data"""
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        signals = np.zeros(len(X_clean))
        
        for t in range(len(X_clean)):
            action = self.act(X_clean[t], training=False)
            signals[t] = (action - 1)
        
        return signals

# ============================================================================
# SECTION 8: EXECUTION & MICROSTRUCTURE MODELS
# ============================================================================

class ExecutionModels:
    """VWAP, TWAP, Smart Order Routing"""
    
    def vwap_execution(self, prices, volumes, signal):
        """VWAP execution logic"""
        vwap = (prices * volumes).rolling(50, min_periods=10).sum() / \
               volumes.rolling(50, min_periods=10).sum()
        execution_quality = 1.0 - np.abs(prices - vwap) / (prices + 1e-8)
        adjusted_signal = signal * execution_quality.fillna(1).values
        return adjusted_signal
    
    def twap_execution(self, signal, window=20):
        """TWAP execution logic"""
        signal_series = pd.Series(signal)
        twap_signal = signal_series.rolling(window, min_periods=5).mean()
        return twap_signal.fillna(0).values
    
    def smart_order_routing(self, signal, spread):
        """Penalize wide spreads"""
        spread_mean = spread.mean()
        if spread_mean == 0:
            spread_mean = 1.0
        spread_penalty = np.exp(-spread / spread_mean)
        signal_series = pd.Series(signal)
        adjusted_signal = signal_series * spread_penalty
        return adjusted_signal.values
    
    def price_impact_model(self, signal, volume):
        """Model price impact and slippage"""
        avg_vol = volume.rolling(100, min_periods=20).mean()
        vol_ratio = volume / (avg_vol + 1e-8)
        impact = np.abs(signal) / (vol_ratio + 1e-8)
        slippage = impact * 0.0001
        return slippage
    
    def apply_all(self, df, signal):
        """Apply execution models as penalties, not averaging"""
        signal_series = pd.Series(signal, index=df.index)
        
        # Execution quality from VWAP
        vwap = (df['price'] * df['volume']).rolling(50, min_periods=10).sum() / \
               df['volume'].rolling(50, min_periods=10).sum()
        execution_quality = 1.0 - np.abs(df['price'] - vwap) / (df['price'] + 1e-8)
        execution_quality = execution_quality.fillna(1.0).clip(0, 1)
        
        # Spread penalty
        if 'LACM_SpreadCost' in df.columns:
            spread = df['LACM_SpreadCost']
        else:
            spread = df['price'] * 0.0001
        
        spread_mean = spread.mean()
        if spread_mean == 0:
            spread_mean = 1.0
        spread_penalty = np.exp(-spread / spread_mean)
        spread_penalty = spread_penalty.clip(0, 1)
        
        # Price impact and slippage
        avg_vol = df['volume'].rolling(100, min_periods=20).mean()
        vol_ratio = df['volume'] / (avg_vol + 1e-8)
        impact = np.abs(signal) / (vol_ratio + 1e-8)
        slippage = impact * 0.0001
        
        # Combined execution adjustment
        final_signal = signal * (
            0.5 * execution_quality.values +
            0.3 * spread_penalty.values +
            0.2
        )
        final_signal = final_signal - slippage.values
        
        return final_signal

class MicrostructureModels:
    """Order book imbalance, queue position, hidden liquidity"""
    
    def fit_all(self, df):
        """Compute microstructure signals"""
        signals = pd.DataFrame(index=df.index)
        
        # Use existing execution cost features if available
        if 'LACM_SpreadCost' in df.columns:
            signals['ob_imbalance'] = -np.tanh(df['LACM_SpreadCost'] / 
                                                (df['LACM_SpreadCost'].mean() + 1e-8))
        else:
            signals['ob_imbalance'] = 0.0
        
        if 'LACM_TradeSize' in df.columns:
            signals['queue_pos'] = np.tanh(df['LACM_TradeSize'] / 
                                           (df['LACM_TradeSize'].mean() + 1e-8))
        else:
            signals['queue_pos'] = 0.0
        
        # Hidden liquidity proxy
        vol_shock = (df['volume'] - df['volume'].rolling(50, min_periods=10).mean()) / \
                    (df['volume'].rolling(50, min_periods=10).std() + 1e-8)
        price_shock = np.abs(df['returns']) / (df['returns'].rolling(50, min_periods=10).std() + 1e-8)
        signals['hidden_liq'] = np.tanh(vol_shock / (price_shock + 1e-8))
        
        return signals.fillna(0)

# ============================================================================
# SECTION 9: ENSEMBLE & MODEL FUSION
# ============================================================================

class EnsembleModel:
    """Combine all model signals with explicit grouping"""
    
    def __init__(self):
        self.weights = {}
        self.meta_model = None
        
    def combine_all(self, signal_groups, method='weighted'):
        """Combine all model outputs using explicit groups"""
        
        # Extract groups
        stat_sig = signal_groups.get('statistical', 0)
        ml_sig = signal_groups.get('machine_learning', 0)
        dl_sig = signal_groups.get('deep_learning', 0)
        rl_sig = signal_groups.get('reinforcement_learning', 0)
        micro_sig = signal_groups.get('microstructure', 0)
        
        # Weighted ensemble
        weights = {
            'stat': 0.15,
            'ml': 0.25,
            'dl': 0.25,
            'rl': 0.20,
            'micro': 0.15
        }
        
        final_signal = (
            weights['stat'] * stat_sig +
            weights['ml'] * ml_sig +
            weights['dl'] * dl_sig +
            weights['rl'] * rl_sig +
            weights['micro'] * micro_sig
        )
        
        return final_signal

# ============================================================================
# SECTION 10: BACKTESTING & EVALUATION
# ============================================================================

class Backtester:
    """Evaluate strategy performance"""
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        
    def backtest(self, signals, returns, transaction_cost=0.0002):
        """Run backtest with soft position sizing"""
        # Use signal magnitude, not just sign
        positions = np.clip(signals, -1, 1)
        strategy_returns = positions[:-1] * returns[1:]
        
        position_changes = np.abs(np.diff(positions))
        costs = position_changes * transaction_cost
        
        net_returns = strategy_returns - costs
        cumulative_returns = np.cumprod(1 + net_returns)
        
        final_value = self.initial_capital * cumulative_returns[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        sharpe = np.mean(net_returns) / (np.std(net_returns) + 1e-8) * np.sqrt(252)
        
        cum_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - cum_max) / (cum_max + 1e-8)
        max_drawdown = np.min(drawdown)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'final_value': final_value,
            'cumulative_returns': cumulative_returns,
            'net_returns': net_returns
        }
    
    def classification_metrics(self, y_true, final_signal):
        """Compute classification metrics with normalized signal"""
        # Normalize signal to [0, 1] range for probability interpretation
        signal_min = final_signal.min()
        signal_max = final_signal.max()
        
        if signal_max - signal_min > 1e-8:
            y_pred_proba = (final_signal - signal_min) / (signal_max - signal_min)
        else:
            y_pred_proba = np.ones_like(final_signal) * 0.5
        
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_true, y_pred_proba)
        except:
            auc = 0.5
        
        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'roc_auc': auc
        }

# ============================================================================
# SECTION 11: MAIN ORCHESTRATION
# ============================================================================

def run_hft_system(filepath):
    """Main execution pipeline with progress tracking"""
    
    print("\n" + "=" * 80)
    print("🚀 HFT QUANTITATIVE TRADING SYSTEM - INSTITUTIONAL DATASET")
    print("=" * 80 + "\n")
    
    # Progress tracker
    total_steps = 12
    
    # Load data
    print(f"[1/{total_steps}] 📂 Loading institutional dataset...")
    df = load_institutional_data(filepath)
    print(f"    ✅ Loaded {len(df)} rows with {df.shape[1]} features\n")
    time.sleep(0.5)
    
    # Feature selection
    print(f"[2/{total_steps}] 🔍 Intelligent feature selection...")
    feature_cols = select_intelligent_features(df)
    
    # Remove non-numeric and target columns
    feature_cols = [c for c in feature_cols if c not in 
                   ['Date', 'Symbol', 'timestamp', 'returns', 'Daily_Return', 
                    'price', 'volume']]
    feature_cols = [c for c in feature_cols if df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
    
    # Additional validation
    valid_cols = []
    for col in feature_cols:
        nan_pct = df[col].isnull().sum() / len(df)
        if nan_pct < 0.5:
            valid_cols.append(col)
    
    feature_cols = valid_cols
    print(f"    ✅ Using {len(feature_cols)} validated features\n")
    time.sleep(0.5)
    
    # Initialize models
    stat_models = StatisticalModels()
    ml_models = MLModels()
    dl_models = DeepLearningModels()
    unsup_models = UnsupervisedModels()
    exec_models = ExecutionModels()
    micro_models = MicrostructureModels()
    ensemble = EnsembleModel()
    backtester = Backtester()
    
    # Statistical models
    print(f"[3/{total_steps}] 📊 Training statistical models...")
    print("    - Mean Reversion (Z-Score)...")
    print("    - Statistical Arbitrage...")
    print("    - ARIMA (rolling window)...")
    print("    - GARCH (rolling window)...")
    print("    - Kalman Filter...")
    stat_signals = stat_models.fit_all(df)
    print(f"    ✅ Generated {stat_signals.shape[1]} statistical signals\n")
    time.sleep(0.5)
    
    # ML models
    print(f"[4/{total_steps}] 🤖 Training ML models...")
    ml_signals, X_train, X_test, y_train, y_test = ml_models.fit_all(df, feature_cols)
    print(f"    ✅ Generated {ml_signals.shape[1]} ML signals\n")
    time.sleep(0.5)
    
    # Deep learning
    print(f"[5/{total_steps}] 🧠 Training deep learning models...")
    dl_signals_dict = dl_models.fit_all(X_train, y_train, X_test, y_test)
    print(f"    ✅ Generated {len(dl_signals_dict)} DL signals\n")
    time.sleep(0.5)
    
    # Unsupervised
    print(f"[6/{total_steps}] 🔬 Training unsupervised models...")
    X_all = df[feature_cols].fillna(0).values
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    unsup_signals = unsup_models.fit_all(X_all)
    print(f"    ✅ Generated {len(unsup_signals)} unsupervised signals\n")
    time.sleep(0.5)
    
    # RL agent
    print(f"[7/{total_steps}] 🎮 Training RL agent (DQN)...")
    split_idx = int(len(X_all) * 0.8)
    X_train_rl = X_all[:split_idx]
    X_test_rl = X_all[split_idx:]
    ret_train_rl = df['returns'].values[:split_idx]
    ret_test_rl = df['returns'].values[split_idx:]
    
    rl_agent = DQNAgent(len(feature_cols))
    rl_agent.train_agent(X_train_rl, ret_train_rl, episodes=5)
    
    rl_signals = np.zeros(len(X_all))
    rl_signals[split_idx:] = rl_agent.predict_signals(X_test_rl)
    print(f"    ✅ RL agent trained (test-only signals, no leakage)\n")
    time.sleep(0.5)
    
    # Microstructure
    print(f"[8/{total_steps}] 💹 Computing microstructure signals...")
    micro_signals = micro_models.fit_all(df)
    print(f"    ✅ Generated {micro_signals.shape[1]} microstructure signals\n")
    time.sleep(0.5)
    
    # Combine signals
    print(f"[9/{total_steps}] 🔗 Combining signals with explicit groups...")
    
    # Statistical group
    stat_group = np.mean([
        stat_signals['mean_rev'].values,
        stat_signals['stat_arb'].values,
        stat_signals['arima'].values,
        stat_signals['garch'].values,
        stat_signals['kalman'].values
    ], axis=0)
    
    # ML group
    ml_group = np.mean([
        ml_signals['svm'].values,
        ml_signals['rf'].values
    ], axis=0)
    
    # DL group
    dl_group = []
    for name, sig in dl_signals_dict.items():
        padded = np.zeros(len(df))
        padded[-len(sig):] = sig
        dl_group.append(padded)
    dl_group = np.mean(dl_group, axis=0) if dl_group else np.zeros(len(df))
    
    # Microstructure group
    micro_group = np.mean([
        micro_signals['ob_imbalance'].values,
        micro_signals['queue_pos'].values,
        micro_signals['hidden_liq'].values
    ], axis=0)
    
    signal_groups = {
        'statistical': stat_group,
        'machine_learning': ml_group,
        'deep_learning': dl_group,
        'reinforcement_learning': rl_signals,
        'microstructure': micro_group
    }
    print(f"    ✅ Grouped 5 signal families\n")
    time.sleep(0.5)
    
    # Ensemble
    print(f"[10/{total_steps}] 🎯 Creating ensemble signal...")
    ensemble_signal = ensemble.combine_all(signal_groups)
    print(f"    ✅ Ensemble signal generated\n")
    time.sleep(0.5)
    
    # Execution models
    print(f"[11/{total_steps}] ⚡ Applying execution models...")
    print("    - VWAP execution quality...")
    print("    - Spread penalty...")
    print("    - Price impact & slippage...")
    final_signal = exec_models.apply_all(df, ensemble_signal)
    print(f"    ✅ Execution-adjusted signal ready\n")
    time.sleep(0.5)
    
    # Backtest
    print(f"[12/{total_steps}] 📈 Running backtest & evaluation...")
    backtest_results = backtester.backtest(final_signal, df['returns'].values)
    
    # Classification metrics
    test_start = int(len(df) * 0.8)
    y_true_test = (df['returns'].iloc[test_start:] > 0).astype(int).values
    
    clf_metrics = backtester.classification_metrics(y_true_test, final_signal[test_start:])
    print(f"    ✅ Backtest complete\n")
    time.sleep(0.5)
    
    # Results
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE - RESULTS")
    print("=" * 80 + "\n")
    
    print("\n📊 BACKTEST PERFORMANCE:")
    print(f"  Total Return:     {backtest_results['total_return']*100:>8.2f}%")
    print(f"  Sharpe Ratio:     {backtest_results['sharpe_ratio']:>8.3f}")
    print(f"  Max Drawdown:     {backtest_results['max_drawdown']*100:>8.2f}%")
    print(f"  Final Portfolio:  ${backtest_results['final_value']:>12,.2f}")
    
    print("\n🎯 CLASSIFICATION METRICS (Test Set):")
    print(f"  Accuracy:         {clf_metrics['accuracy']:>8.3f}")
    print(f"  Precision:        {clf_metrics['precision']:>8.3f}")
    print(f"  Recall:           {clf_metrics['recall']:>8.3f}")
    print(f"  ROC-AUC:          {clf_metrics['roc_auc']:>8.3f}")
    
    print("\n🤖 MODEL CONTRIBUTIONS:")
    print(f"  Statistical:      5 signals (rolling ARIMA/GARCH)")
    print(f"  ML (SVM, RF):     2 signals")
    print(f"  DL (MLP, LSTM):   {len(dl_signals_dict)} signals")
    print(f"  RL (DQN):         1 signal (test-only, no leakage)")
    print(f"  Microstructure:   3 signals")
    print(f"  Total Groups:     5")
    
    # Plots
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Price
    axes[0].plot(df.index, df['price'].values, label='Price', alpha=0.7)
    axes[0].set_title('Price Series', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Final signal
    axes[1].plot(df.index, final_signal, label='Ensemble Signal', color='red', alpha=0.7)
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[1].set_title('Final Trading Signal', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Cumulative returns
    axes[2].plot(df.index[1:], backtest_results['cumulative_returns'], 
                label='Strategy Returns', color='green', linewidth=2)
    axes[2].set_title('Cumulative Returns', fontsize=12, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Model comparison
    signal_comparison = pd.DataFrame({
        'Statistical': stat_group,
        'ML': ml_group,
        'DL': dl_group,
        'RL': rl_signals,
        'Microstructure': micro_group
    })
    signal_comparison.plot(ax=axes[3], alpha=0.6)
    axes[3].set_title('Model Signal Comparison', fontsize=12, fontweight='bold')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hft_institutional_results.png', dpi=150, bbox_inches='tight')
    print("\n📈 Results plot saved: 'hft_institutional_results.png'")
    
    # Signal interpretation
    print("\n💡 CURRENT TRADING SIGNAL:")
    current_signal = final_signal[-1]
    if current_signal > 0.3:
        print(f"  🟢 STRONG BUY (signal: {current_signal:.3f})")
    elif current_signal > 0.1:
        print(f"  🟢 BUY (signal: {current_signal:.3f})")
    elif current_signal < -0.3:
        print(f"  🔴 STRONG SELL (signal: {current_signal:.3f})")
    elif current_signal < -0.1:
        print(f"  🔴 SELL (signal: {current_signal:.3f})")
    else:
        print(f"  🟡 HOLD (signal: {current_signal:.3f})")
    
    # Leverage existing features
    if 'FinalScore' in df.columns:
        existing_correlation = np.corrcoef(df['FinalScore'].fillna(0), final_signal)[0,1]
        print(f"\n📊 Correlation with existing FinalScore: {existing_correlation:.3f}")
    
    print("\n" + "=" * 80)
    print("✅ SYSTEM EXECUTION COMPLETE - ALL MODELS TRAINED")
    print("=" * 80 + "\n")
    
    return {
        'df': df,
        'signal_groups': signal_groups,
        'final_signal': final_signal,
        'backtest': backtest_results,
        'metrics': clf_metrics,
        'feature_cols': feature_cols
    }

# ============================================================================
# EXECUTION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # ========================================================================
    # CONFIGURATION: UPDATE YOUR FILENAME HERE
    # ========================================================================
    EXCEL_FILE_PATH = "Enhanced_Portfolio_AllAlgorithms_20251113_124034 (1).xlsx"
    # ========================================================================
    
    print("=" * 80)
    print("HFT QUANTITATIVE TRADING SYSTEM")
    print("=" * 80)
    print(f"\nTarget file: {EXCEL_FILE_PATH}\n")
    
    try:
        results = run_hft_system(EXCEL_FILE_PATH)
        print("\n✅ Training complete!")
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: File '{EXCEL_FILE_PATH}' not found!")
        print("\nPlease ensure:")
        print("1. Your Excel file is uploaded to Colab (use folder icon on left)")
        print("2. The filename in EXCEL_FILE_PATH matches exactly (including spaces)")
        print("3. The file is in the current directory")
        print("\nTo upload: Click folder icon → Upload button → Select your .xlsx file")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
