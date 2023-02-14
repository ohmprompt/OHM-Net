import tensorflow as tf
import tensorflow_probability as tfp

# Load image
img_path = "path/to/image.jpg"
img_raw = tf.io.read_file(img_path)
img = tf.image.decode_image(img_raw)

# Convert to grayscale
img_gray = tf.image.rgb_to_grayscale(img)

# Flatten image
img_flat = tf.reshape(img_gray, [-1])

# Initialize GMM parameters
num_components = 5
means = tf.Variable(tf.random.normal(shape=[num_components], mean=0.0, stddev=1.0))
covs = tf.Variable(tf.random.uniform(shape=[num_components], minval=0.0, maxval=1.0))
weights = tf.Variable(tf.ones(shape=[num_components]) / num_components)

# Define log-likelihood function
def log_likelihood(x):
  return tf.reduce_sum([tfd.Normal(loc=means[k], scale=tf.sqrt(covs[k])).log_prob(x) * weights[k] for k in range(num_components)])

# Implement EM algorithm
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

for i in range(100):
  with tf.GradientTape() as tape:
    log_likelihood_value = log_likelihood(img_flat)
    loss = -log_likelihood_value
  grads = tape.gradient(loss, [means, covs, weights])
  optimizer.apply_gradients(zip(grads, [means, covs, weights]))

# Find background
sorted_means = tf.sort(means)
background_mean = sorted_means[0]

# Visualize background
background_flat = tf.ones_like(img_flat) * background_mean
background = tf.reshape(background_flat, tf.shape(img_gray))
background = tf.image.convert_image_dtype(background, tf.uint8)
background_raw = tf.image.encode_png(background)

# Write background image to file
background_path = "path/to/background.png"
tf.io.write_file(background_path, background_raw)
