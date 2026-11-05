"""
EXPERIMENT 8: RNN with LSTM
=========================================
Objectives:
    - Understand LSTM architecture and gates
    - Implement LSTM for sequence prediction
    - Compare with simple RNN
    - Handle longer sequences

Dataset:
    Synthetic stock price data (trend + noise)

LSTM Architecture:
    Input (60 timesteps, 1 feature)
        -> LSTM (100, return_sequences=True) + Dropout (0.2)
        -> LSTM (50) + Dropout (0.2)
        -> Dense (25, ReLU)
        -> Dense (1) - Output

Expected Outcomes:
    - LSTM RMSE: ~0.05 (better than RNN)
    - LSTM handles long sequences better
    - LSTM training more stable
    - Clear advantage demonstrated
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, SimpleRNN, Dense, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler


# Generate more complex time series
def generate_stock_prices(num_samples=500, seq_length=60):
    """Generate synthetic stock price data with trend and noise"""
    prices = []
    sequences = []

    # Generate price with trend
    base_price = 100
    trend = np.random.normal(0.001, 0.01, num_samples + seq_length)
    price_series = base_price * np.exp(np.cumsum(trend))

    for i in range(num_samples):
        sequences.append(price_series[i:i + seq_length])
        prices.append(price_series[i + seq_length])

    return np.array(sequences), np.array(prices)


# Generate data
X, y = generate_stock_prices(num_samples=500, seq_length=60)

# Normalize
scaler = MinMaxScaler()
X_normalized = np.zeros_like(X)
for i in range(len(X)):
    X_normalized[i] = scaler.fit_transform(X[i].reshape(-1, 1)).flatten()

y_normalized = scaler.fit_transform(y.reshape(-1, 1)).flatten()

# Split data
split_idx = 400
X_train = X_normalized[:split_idx].reshape(split_idx, 60, 1)
X_test = X_normalized[split_idx:].reshape(len(X_normalized) - split_idx, 60, 1)
y_train = y_normalized[:split_idx]
y_test = y_normalized[split_idx:]

print(f"Training shape: {X_train.shape}, {y_train.shape}")
print(f"Test shape: {X_test.shape}, {y_test.shape}")

# Build LSTM model
model_lstm = Sequential([
    LSTM(100, activation='relu', return_sequences=True, input_shape=(60, 1)),
    Dropout(0.2),
    LSTM(50, activation='relu', return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

model_lstm.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

model_lstm.summary()

# Train LSTM
print("\nTraining LSTM...")
history_lstm = model_lstm.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Predictions
y_pred_lstm = model_lstm.predict(X_test, verbose=0).flatten()

# For comparison, build Simple RNN
model_rnn = Sequential([
    SimpleRNN(100, activation='relu', return_sequences=True, input_shape=(60, 1)),
    Dropout(0.2),
    SimpleRNN(50, activation='relu', return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

model_rnn.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("\nTraining Simple RNN...")
history_rnn = model_rnn.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

y_pred_rnn = model_rnn.predict(X_test, verbose=0).flatten()

# Compare
rmse_lstm = np.sqrt(np.mean((y_test - y_pred_lstm) ** 2))
rmse_rnn = np.sqrt(np.mean((y_test - y_pred_rnn) ** 2))

print("\n" + "=" * 40)
print(f"LSTM RMSE:       {rmse_lstm:.4f}")
print(f"Simple RNN RMSE: {rmse_rnn:.4f}")
print("=" * 40)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Plot 1: Loss comparison
axes[0, 0].plot(history_lstm.history['val_loss'], label='LSTM')
axes[0, 0].plot(history_rnn.history['val_loss'], label='Simple RNN')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Validation Loss')
axes[0, 0].set_title('Validation Loss Comparison')
axes[0, 0].legend()
axes[0, 0].grid()

# Plot 2: MAE comparison
axes[0, 1].plot(history_lstm.history['val_mae'], label='LSTM')
axes[0, 1].plot(history_rnn.history['val_mae'], label='Simple RNN')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Validation MAE')
axes[0, 1].set_title('Validation MAE Comparison')
axes[0, 1].legend()
axes[0, 1].grid()

# Plot 3: LSTM predictions
axes[1, 0].plot(y_test, label='Actual')
axes[1, 0].plot(y_pred_lstm, label='LSTM Prediction')
axes[1, 0].set_xlabel('Sample')
axes[1, 0].set_ylabel('Normalized Price')
axes[1, 0].set_title(f'LSTM Predictions (RMSE = {rmse_lstm:.4f})')
axes[1, 0].legend()
axes[1, 0].grid()

# Plot 4: RNN predictions
axes[1, 1].plot(y_test, label='Actual')
axes[1, 1].plot(y_pred_rnn, label='RNN Prediction')
axes[1, 1].set_xlabel('Sample')
axes[1, 1].set_ylabel('Normalized Price')
axes[1, 1].set_title(f'Simple RNN Predictions (RMSE = {rmse_rnn:.4f})')
axes[1, 1].legend()
axes[1, 1].grid()

plt.tight_layout()
plt.show()
