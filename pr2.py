import time
import tensorflow as tf

# Create large tensor
x = tf.random.normal([10000, 10000])

# Measure computation time
start = time.time()
y = tf.matmul(x, x)
end = time.time()

print(f"Computation time: {end - start:.4f} seconds")
