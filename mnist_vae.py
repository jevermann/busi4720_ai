# From the example at https://keras.io/examples/generative/vae/
#

import tensorflow as tf
import keras
import numpy as np
from tensorflow.keras import layers

# Sampling Layer
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Encoder (not a sequential model)
encoder_inputs = tf.keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, (3, 3), activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, (3, 3), activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
# Vector of means
z_mean = layers.Dense(10, name="z_mean")(x)
# Vector of log variances
z_log_var = layers.Dense(10, name="z_log_var")(x)
# Point in latent space
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z])

# Decoder
latent_inputs = keras.Input(shape=(10,))
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

# VAE Model
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.vae_loss_tracker = keras.metrics.Mean(name="total_loss")

    # Loss function
    def vae_loss(self, encoder, decoder, data):
        z_mean, z_log_var, z = encoder(data)
        reconstruction = decoder(z)
        mse_loss = keras.losses.MeanSquaredError()
        reconstruction_loss = mse_loss(data, reconstruction)
        kl_loss = -0.5 * (1 +
                          z_log_var -
                          tf.square(z_mean) -
                          tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss
        return total_loss
        
	# Training step for one batch
    def train_step(self, data):
        with tf.GradientTape() as tape:
            total_loss = self.vae_loss(self.encoder, self.decoder, data)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.vae_loss_tracker.update_state(total_loss)
        return {"loss": self.vae_loss_tracker.result()}

# Load data, combine train and test
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
# Transform inputs to floats in [0,1]
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

# Define an instance of the the VAE model class
vae = VAE(encoder, decoder)
# Compile and fit
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(mnist_digits, epochs=30, batch_size=128)
