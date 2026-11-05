"""
EXPERIMENT 9: RNN with GRU
=========================================
Objectives:
    - Understand GRU architecture (simpler than LSTM)
    - Implement GRU for sequence tasks
    - Compare GRU, LSTM, and RNN
    - Analyze computational efficiency

Dataset:
    Synthetic stock price time series (same as Experiment 8)

GRU vs LSTM vs RNN:
    Aspect               RNN        LSTM       GRU
    Vanishing Gradient   Poor       Good       Good
    Parameters           Fewest     Most       Medium
    Training Speed       Fast       Slow       Medium
    Long Sequences       Poor       Good       Good
    Complexity           Low        High       Medium

Expected Outcomes:
    - GRU performance similar to LSTM
    - GRU trains faster than LSTM
    - All handle longer sequences better than basic RNN
    - Clear performance/efficiency tradeoff visualization
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout, LSTM, SimpleRNN
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler


# Use same data as Experiment 8
def generate_stock_prices(num_samples=500, seq_length=60):
    prices = []
    sequences = []
    base_price = 100
    trend = np.random.normal(0.001, 0.01, num_samples + seq_length)
    price_series = base_price * np.exp(np.cumsum(trend))
    for i in range(num_samples):
        sequences.append(price_series[i:i + seq_length])
        prices.append(price_series[i + seq_length])
    return np.array(sequences), np.array(prices)


X, y = generate_stock_prices(num_samples=500, seq_length=60)

scaler = MinMaxScaler()
X_normalized = np.zeros_like(X)
for i in range(len(X)):
    X_normalized[i] = scaler.fit_transform(X[i].reshape(-1, 1)).flatten()
y_normalized = scaler.fit_transform(y.reshape(-1, 1)).flatten()

split_idx = 400
X_train = X_normalized[:split_idx].reshape(split_idx, 60, 1)
X_test = X_normalized[split_idx:].reshape(len(X_normalized) - split_idx, 60, 1)
y_train = y_normalized[:split_idx]
y_test = y_normalized[split_idx:]


def build_model(layer_cls):
    """Build the same architecture with a different recurrent layer."""
    model = Sequential([
        layer_cls(100, activation='relu', return_sequences=True, input_shape=(60, 1)),
        Dropout(0.2),
        layer_cls(50, activation='relu', return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


def train_and_time(model, name):
    print(f"\nTraining {name}...")
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    train_time = time.time() - start_time
    y_pred = model.predict(X_test, verbose=0).flatten()
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    print(f"{name}: RMSE = {rmse:.4f}, Time = {train_time:.1f}s, Params = {model.count_params():,}")
    return history, y_pred, rmse, train_time


# GRU Model
print("Building GRU model...")
model_gru = build_model(GRU)
model_gru.summary()
history_gru, y_pred_gru, rmse_gru, gru_train_time = train_and_time(model_gru, "GRU")

# LSTM for comparison (from Exp 8)
print("\nBuilding LSTM model for comparison...")
model_lstm = build_model(LSTM)
history_lstm, y_pred_lstm, rmse_lstm, lstm_train_time = train_and_time(model_lstm, "LSTM")

# Simple RNN for comparison
print("\nBuilding Simple RNN model for comparison...")
model_rnn = build_model(SimpleRNN)
history_rnn, y_pred_rnn, rmse_rnn, rnn_train_time = train_and_time(model_rnn, "Simple RNN")

# Summary table
print("\n" + "=" * 55)
print(f"{'Model':<12}{'RMSE':<12}{'Time (s)':<12}{'Parameters':<12}")
print("-" * 55)
print(f"{'Simple RNN':<12}{rmse_rnn:<12.4f}{rnn_train_time:<12.1f}{model_rnn.count_params():<12,}")
print(f"{'LSTM':<12}{rmse_lstm:<12.4f}{lstm_train_time:<12.1f}{model_lstm.count_params():<12,}")
print(f"{'GRU':<12}{rmse_gru:<12.4f}{gru_train_time:<12.1f}{model_gru.count_params():<12,}")
print("=" * 55)

# Visualization
names = ['Simple RNN', 'LSTM', 'GRU']
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Plot 1: Validation loss
axes[0, 0].plot(history_rnn.history['val_loss'], label='Simple RNN')
axes[0, 0].plot(history_lstm.history['val_loss'], label='LSTM')
axes[0, 0].plot(history_gru.history['val_loss'], label='GRU')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Validation Loss')
axes[0, 0].set_title('Validation Loss Comparison')
axes[0, 0].legend()
axes[0, 0].grid()

# Plot 2: RMSE
axes[0, 1].bar(names, [rmse_rnn, rmse_lstm, rmse_gru], color=['tab:red', 'tab:blue', 'tab:green'])
axes[0, 1].set_ylabel('Test RMSE')
axes[0, 1].set_title('Test RMSE (lower is better)')
axes[0, 1].grid(axis='y')

# Plot 3: Training time
axes[1, 0].bar(names, [rnn_train_time, lstm_train_time, gru_train_time],
               color=['tab:red', 'tab:blue', 'tab:green'])
axes[1, 0].set_ylabel('Training Time (s)')
axes[1, 0].set_title('Training Time (lower is faster)')
axes[1, 0].grid(axis='y')

# Plot 4: Predictions
axes[1, 1].plot(y_test, label='Actual', color='black')
axes[1, 1].plot(y_pred_rnn, label='Simple RNN', alpha=0.6)
axes[1, 1].plot(y_pred_lstm, label='LSTM', alpha=0.6)
axes[1, 1].plot(y_pred_gru, label='GRU', alpha=0.6)
axes[1, 1].set_xlabel('Sample')
axes[1, 1].set_ylabel('Normalized Price')
axes[1, 1].set_title('Predictions vs Actual')
axes[1, 1].legend()
axes[1, 1].grid()

plt.tight_layout()
plt.show()
