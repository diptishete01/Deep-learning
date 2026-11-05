"""
EXPERIMENT 11: Neural Style Transfer
=========================================
Objectives:
    - Understand style transfer concept
    - Implement Gram matrix technique
    - Separate content and style features
    - Apply artistic styles to images

Prerequisites:
    - Pillow library for image handling
    - Pre-trained VGG19 model

Style Transfer Pipeline:
    Content Image -> VGG19 -> Extract Deep Features (Content)
    Style Image   -> VGG19 -> Extract & Compute Gram Matrices (Style)
    Generated Image (initialized from content)
        -> Optimize to minimize:
           |Content_features - Generated_content_features|^2
           + lambda * |Gram_style - Gram_generated|^2
        -> Output: Stylized Image

Expected Outcomes:
    - Understanding of Gram matrices for style
    - Separation of style and content features
    - Ability to implement style transfer algorithm
    - Visualized transformation process
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import vgg19
from tensorflow.keras.models import Model
import urllib.request

# ---------------- Settings ----------------
CONTENT_PATH = "content.jpg"
STYLE_PATH = "style.jpg"
MAX_DIM = 400
STEPS = 150            # lower this (e.g. 50) if it is too slow on your laptop
CONTENT_WEIGHT = 1e4
STYLE_WEIGHT = 1e-2
TV_WEIGHT = 30         # smoothness term, reduces noise in the result

# Sample images (used only if you don't provide your own content.jpg / style.jpg)
SAMPLE_CONTENT_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg"
SAMPLE_STYLE_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg"


def ensure_image(path, url):
    if not os.path.exists(path):
        print(f"Downloading sample image to {path}...")
        urllib.request.urlretrieve(url, path)


# Load pre-trained VGG19
def load_vgg19():
    vgg = vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    return vgg


# Preprocess image
def load_and_process_image(image_path, max_dim=400):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((max_dim, max_dim))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Deprocess image
def deprocess_image(img_array):
    img = np.array(img_array).squeeze()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype('uint8')
    return Image.fromarray(img)


# Gram matrix (style representation)
def gram_matrix(x):
    """Calculate Gram matrix for style representation"""
    x = tf.squeeze(x, axis=0)                 # (h, w, channels)
    h = tf.shape(x)[0]
    w = tf.shape(x)[1]
    x = tf.transpose(x, (2, 0, 1))            # (channels, h, w)
    n = tf.shape(x)[0]
    x = tf.reshape(x, (n, -1))                # (channels, h*w)
    gram = tf.linalg.matmul(x, tf.transpose(x))
    return gram / tf.cast(h * w, tf.float32)  # normalize by image size


# Build style transfer model
def get_style_transfer_model():
    vgg = load_vgg19()

    # Use specific layers for content and style
    content_layers = ['block5_conv2']  # Deep layer for content
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']  # Multiple layers for style

    layer_outputs = [vgg.get_layer(name).output for name in content_layers + style_layers]
    model = Model(inputs=vgg.input, outputs=layer_outputs)
    model.trainable = False

    return model, len(content_layers), len(style_layers)


# Loss functions
def content_loss(content_actual, content_generated):
    return tf.reduce_mean(tf.square(content_actual - content_generated))


def style_loss(style_actual, style_generated):
    """MSE of Gram matrices"""
    gram_actual = gram_matrix(style_actual)
    gram_generated = gram_matrix(style_generated)
    return tf.reduce_mean(tf.square(gram_actual - gram_generated))


def extract_features(model, img, num_content):
    """img is in [0, 1]; VGG19 needs its own preprocessing."""
    x = vgg19.preprocess_input(img * 255.0)
    outputs = model(x)
    return outputs[:num_content], outputs[num_content:]


# ---------------- Run style transfer ----------------
ensure_image(CONTENT_PATH, SAMPLE_CONTENT_URL)
ensure_image(STYLE_PATH, SAMPLE_STYLE_URL)

content_img = load_and_process_image(CONTENT_PATH, MAX_DIM)
style_img = load_and_process_image(STYLE_PATH, MAX_DIM)

model, num_content, num_style = get_style_transfer_model()

# Target features from the content and style images
target_content, _ = extract_features(model, tf.constant(content_img), num_content)
_, target_style = extract_features(model, tf.constant(style_img), num_content)

# Generated image starts as a copy of the content image and is optimized
generated = tf.Variable(content_img)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)


def train_step():
    with tf.GradientTape() as tape:
        gen_content, gen_style = extract_features(model, generated, num_content)

        c_loss = tf.add_n([content_loss(t, g) for t, g in zip(target_content, gen_content)])
        c_loss = c_loss * CONTENT_WEIGHT / num_content

        s_loss = tf.add_n([style_loss(t, g) for t, g in zip(target_style, gen_style)])
        s_loss = s_loss * STYLE_WEIGHT / num_style

        tv_loss = tf.reduce_sum(tf.image.total_variation(generated)) * TV_WEIGHT

        total = c_loss + s_loss + tv_loss

    grads = tape.gradient(total, generated)
    optimizer.apply_gradients([(grads, generated)])
    generated.assign(tf.clip_by_value(generated, 0.0, 1.0))
    return total, c_loss, s_loss


print("\nRunning style transfer...")
snapshots = {}
loss_history = []
for step in range(1, STEPS + 1):
    total, c_loss, s_loss = train_step()
    loss_history.append(float(total))
    if step % 10 == 0 or step == 1:
        print(f"Step {step}/{STEPS} - total: {float(total):.1f}, "
              f"content: {float(c_loss):.1f}, style: {float(s_loss):.1f}")
    if step in (STEPS // 3, 2 * STEPS // 3, STEPS):
        snapshots[step] = generated.numpy().copy()

result = deprocess_image(generated.numpy())
result.save("stylized_output.png")
print("Saved stylized_output.png")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

axes[0, 0].imshow(deprocess_image(content_img))
axes[0, 0].set_title('Content Image')
axes[0, 1].imshow(deprocess_image(style_img))
axes[0, 1].set_title('Style Image')
axes[0, 2].imshow(result)
axes[0, 2].set_title('Stylized Result')

for ax, (step, snap) in zip(axes[1, :2], list(snapshots.items())[:2]):
    ax.imshow(deprocess_image(snap))
    ax.set_title(f'After {step} steps')

axes[1, 2].plot(loss_history)
axes[1, 2].set_xlabel('Step')
axes[1, 2].set_ylabel('Total Loss')
axes[1, 2].set_title('Loss over Optimization')
axes[1, 2].grid()

for ax in axes.flat[:5]:
    ax.axis('off')

plt.tight_layout()
plt.show()
