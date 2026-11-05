"""
EXPERIMENT 12: Image Generation Models (VAE)
=========================================
Objectives:
    - Understand generative models
    - Build and train GAN or VAE (this file: VAE)
    - Generate novel images
    - Analyze generated outputs

Dataset:
    MNIST (28x28 grayscale digits)

VAE Architecture:
    Encoder: Input (28x28x1) -> Conv2D -> Conv2D -> Flatten -> Dense (latent)
    Latent Space: 20D Gaussian (mean, log variance)
    Sampling: z = mean + sigma * epsilon
    Decoder: Dense -> Reshape -> Conv2DTranspose -> Conv2DTranspose -> Output (28x28x1)
    Loss = Reconstruction Loss + KL Divergence

Expected Outcomes:
    - VAE generates reasonable MNIST-like digits
    - Smooth interpolation in latent space
    - Reconstruction error decreases over epochs
    - Understanding of generative modeling
    - Ability to sample novel images
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# Load and prepare MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

print(f"Training data shape: {X_train.shape}")

# Hyperparameters
latent_dim = 20
batch_size = 32
epochs = 30  # lower this (e.g. 10) if it is too slow on your laptop


# Sampling layer (reparameterization trick)
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# Encoder
encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_inputs)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
shape_before_flatten = x.shape[1:]  # (14, 14, 64)
x = layers.Flatten()(x)
z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
z = Sampling(name='z')([z_mean, z_log_var])

encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

# Decoder
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(int(np.prod(shape_before_flatten)))(latent_inputs)
x = layers.Reshape(shape_before_flatten)(x)
x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
decoder_outputs = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder')
decoder.summary()


# VAE: reconstruction loss + KL divergence
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def compute_losses(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)

        # Reconstruction loss: binary crossentropy summed over all pixels
        recon_loss = tf.reduce_mean(
            tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2))
        )
        # KL divergence between q(z|x) and the standard normal prior
        kl_loss = tf.reduce_mean(
            tf.reduce_sum(-0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1)
        )
        return recon_loss + kl_loss, recon_loss, kl_loss

    def _update_metrics(self, total, recon, kl):
        self.total_loss_tracker.update_state(total)
        self.recon_loss_tracker.update_state(recon)
        self.kl_loss_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        if isinstance(data, (tuple, list)):
            data = data[0]
        with tf.GradientTape() as tape:
            total, recon, kl = self.compute_losses(data)
        grads = tape.gradient(total, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return self._update_metrics(total, recon, kl)

    def test_step(self, data):
        if isinstance(data, (tuple, list)):
            data = data[0]
        total, recon, kl = self.compute_losses(data)
        return self._update_metrics(total, recon, kl)


vae = VAE(encoder, decoder, name='vae')
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))

# Train
print("\nTraining VAE...")
history = vae.fit(
    X_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_test, X_test),
    verbose=1
)

# ---------------- Visualization ----------------

# 1. Training curves
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].plot(history.history['loss'], label='Train')
axes[0].plot(history.history['val_loss'], label='Validation')
axes[0].set_title('Total Loss')
axes[1].plot(history.history['reconstruction_loss'], label='Train')
axes[1].plot(history.history['val_reconstruction_loss'], label='Validation')
axes[1].set_title('Reconstruction Loss')
axes[2].plot(history.history['kl_loss'], label='Train')
axes[2].plot(history.history['val_kl_loss'], label='Validation')
axes[2].set_title('KL Divergence')
for ax in axes:
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid()
plt.tight_layout()
plt.show()

# 2. Original vs reconstructed
_, _, z_test = encoder.predict(X_test[:10], verbose=0)
reconstructed = decoder.predict(z_test, verbose=0)

fig, axes = plt.subplots(2, 10, figsize=(16, 3.5))
for i in range(10):
    axes[0, i].imshow(X_test[i].reshape(28, 28), cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].imshow(reconstructed[i].reshape(28, 28), cmap='gray')
    axes[1, i].axis('off')
axes[0, 0].set_title('Original', loc='left')
axes[1, 0].set_title('Reconstructed', loc='left')
plt.tight_layout()
plt.show()

# 3. Generate new digits by sampling from the latent space
z_samples = np.random.normal(size=(20, latent_dim))
generated = decoder.predict(z_samples, verbose=0)

fig, axes = plt.subplots(2, 10, figsize=(16, 3.5))
for i, ax in enumerate(axes.flat):
    ax.imshow(generated[i].reshape(28, 28), cmap='gray')
    ax.axis('off')
fig.suptitle('Newly Generated Digits (sampled from latent space)')
plt.tight_layout()
plt.show()

# 4. Latent space interpolation between two test images
z_start, z_end = z_test[0], z_test[1]
steps = 10
interpolated = np.array([z_start + (z_end - z_start) * t for t in np.linspace(0, 1, steps)])
interp_images = decoder.predict(interpolated, verbose=0)

fig, axes = plt.subplots(1, steps, figsize=(16, 2))
for i in range(steps):
    axes[i].imshow(interp_images[i].reshape(28, 28), cmap='gray')
    axes[i].axis('off')
fig.suptitle('Latent Space Interpolation')
plt.tight_layout()
plt.show()
