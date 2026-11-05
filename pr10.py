"""
EXPERIMENT 10: Text Generation using LSTM
=========================================
Objectives:
    - Understand character-level language modeling
    - Build LSTM for text generation
    - Train on text corpus
    - Generate novel text sequences

Dataset:
    Shakespeare text (Project Gutenberg) or any text corpus

Expected Outcomes:
    - Model learns character patterns from text
    - Generates readable sequences
    - Temperature parameter controls creativity
    - Loss decreases over training
    - Can generate novel (if imperfect) text
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import urllib.request

SEQ_LENGTH = 40
MAX_CHARS = 200_000  # full Shakespeare (~5M chars) needs many GB of RAM

# Download text data
print("Downloading text data...")
url = "https://www.gutenberg.org/cache/epub/100/pg100.txt"
try:
    urllib.request.urlretrieve(url, "shakespeare.txt")
    with open("shakespeare.txt", encoding='utf-8', errors='ignore') as f:
        text = f.read()
except Exception:
    # Fallback: create sample text
    text = """To be or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles,
And by opposing end them. """ * 100

text = text[:MAX_CHARS]
print(f"Text length: {len(text)} characters")

# Create vocabulary
unique_chars = sorted(list(set(text)))
char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}
vocab_size = len(unique_chars)

print(f"Unique characters: {vocab_size}")
print(f"Character mapping: {char_to_idx}")


# Create sequences
def create_sequences(text, char_to_idx, seq_length=40):
    X = []
    y = []
    for i in range(len(text) - seq_length):
        seq = text[i:i + seq_length]
        target = text[i + seq_length]
        X.append([char_to_idx[char] for char in seq])
        y.append(char_to_idx[target])
    return np.array(X), np.array(y)


X, y = create_sequences(text, char_to_idx, seq_length=SEQ_LENGTH)
print(f"Dataset shape - X: {X.shape}, y: {y.shape}")

# Normalize to 0-1 and add feature dimension: (samples, timesteps, 1)
X_normalized = (X / vocab_size).astype('float32').reshape(len(X), SEQ_LENGTH, 1)

# One-hot encode targets
y_categorical = np.zeros((len(y), vocab_size), dtype='float32')
for i, target in enumerate(y):
    y_categorical[i, target] = 1

# Split train-test
split_idx = int(0.8 * len(X))
X_train = X_normalized[:split_idx]
X_test = X_normalized[split_idx:]
y_train = y_categorical[:split_idx]
y_test = y_categorical[split_idx:]

print(f"Training set: {X_train.shape}, {y_train.shape}")
print(f"Test set: {X_test.shape}, {y_test.shape}")

# Build text generation LSTM
model = Sequential([
    LSTM(128, activation='relu', input_shape=(SEQ_LENGTH, 1), return_sequences=True),
    Dropout(0.2),
    LSTM(64, activation='relu'),
    Dropout(0.2),
    Dense(vocab_size, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train
print("\nTraining text generation model...")
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")


# Text generation with temperature
def sample_with_temperature(probs, temperature=1.0):
    probs = np.asarray(probs).astype('float64')
    probs = np.log(probs + 1e-8) / temperature
    probs = np.exp(probs)
    probs = probs / np.sum(probs)
    return np.random.choice(len(probs), p=probs)


def generate_text(seed_text, length=300, temperature=1.0):
    generated = seed_text
    current = seed_text
    for _ in range(length):
        # Encode the last SEQ_LENGTH characters
        encoded = [char_to_idx.get(c, 0) for c in current[-SEQ_LENGTH:]]
        x = np.array(encoded, dtype='float32').reshape(1, SEQ_LENGTH, 1) / vocab_size
        probs = model.predict(x, verbose=0)[0]
        next_char = idx_to_char[sample_with_temperature(probs, temperature)]
        generated += next_char
        current += next_char
    return generated


# Pick a random seed from the text
start = np.random.randint(0, len(text) - SEQ_LENGTH - 1)
seed = text[start:start + SEQ_LENGTH]

for temp in [0.2, 0.5, 1.0]:
    print("\n" + "=" * 60)
    print(f"Temperature: {temp}")
    print("=" * 60)
    print(generate_text(seed, length=300, temperature=temp))

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'], label='Training Loss')
axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Model Loss')
axes[0].legend()
axes[0].grid()

axes[1].plot(history.history['accuracy'], label='Training Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Model Accuracy')
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.show()
