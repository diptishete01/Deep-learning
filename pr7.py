"""
EXPERIMENT 7: Simple RNN
=========================
Objectives:
    - Understand RNN architecture and sequence processing
    - Build simple RNN from scratch
    - Train on sequential data (time series)
    - Visualize hidden states

Dataset:
    Synthetic sine wave time series

RNN Architecture:
    Input Sequence (50 timesteps, 1 feature)
        -> SimpleRNN (64 units, return_sequences=True)
        -> SimpleRNN (32 units)
        -> Dropout (0.2)
        -> Dense (16, ReLU)
        -> Dense (1) - Output

Expected Outcomes:
    - Low RMSE on test set (<0.15)
    - High correlation between predictions and actual values
    - Model captures temporal dependencies
    - Understanding of sequence processing
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Dropout
from keras.optimizers import Adam


# Generate synthetic time series data
def generate_time_series(num_samples=1000, seq_length=50):
    """Generate sine wave time series"""
    data = []
    targets = []
    for _ in range(num_samples):
        start = np.random.uniform(0, 2 * np.pi)
        x = np.sin(np.linspace(start, start + 10 * np.pi, seq_length))
        data.append(x)
        targets.append(x[-1])  # Predict last value
    return np.array(data), np.array(targets)


# Generate data
X, y = generate_time_series(num_samples=1000, seq_length=50)
X = X.reshape(X.shape[0], X.shape[1], 1)  # (samples, timesteps, features)

print(f"Input shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split train-test
split_idx = 800
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Build RNN
model = Sequential([
    SimpleRNN(64, activation='relu', input_shape=(50, 1), return_sequences=True),
    SimpleRNN(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)  # Regression output
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

model.summary()

# Train
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)

print(f"Training Loss: {train_loss[0]:.6f}")
print(f"Test Loss: {test_loss[0]:.6f}")

# Make predictions
y_pred = model.predict(X_test, verbose=0).flatten()

rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
correlation = np.corrcoef(y_test, y_pred)[0, 1]
print(f"Test RMSE: {rmse:.4f}")
print(f"Correlation: {correlation:.4f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Plot 1: Training history
axes[0, 0].plot(history.history['loss'], label='Training Loss')
axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Model Loss')
axes[0, 0].legend()
axes[0, 0].grid()

# Plot 2: MAE
axes[0, 1].plot(history.history['mae'], label='Training MAE')
axes[0, 1].plot(history.history['val_mae'], label='Validation MAE')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('MAE')
axes[0, 1].set_title('Mean Absolute Error')
axes[0, 1].legend()
axes[0, 1].grid()

# Plot 3: Predictions vs Actual
axes[1, 0].plot(y_test[:100], label='Actual', marker='o', markersize=3)
axes[1, 0].plot(y_pred[:100], label='Predicted', marker='x', markersize=3)
axes[1, 0].set_xlabel('Sample')
axes[1, 0].set_ylabel('Value')
axes[1, 0].set_title('Predictions vs Actual')
axes[1, 0].legend()
axes[1, 0].grid()

# Plot 4: Scatter of predicted vs actual
axes[1, 1].scatter(y_test, y_pred, alpha=0.6)
axes[1, 1].plot([-1, 1], [-1, 1], 'r--', label='Perfect prediction')
axes[1, 1].set_xlabel('Actual')
axes[1, 1].set_ylabel('Predicted')
axes[1, 1].set_title(f'Actual vs Predicted (r = {correlation:.3f})')
axes[1, 1].legend()
axes[1, 1].grid()

plt.tight_layout()
plt.show()
