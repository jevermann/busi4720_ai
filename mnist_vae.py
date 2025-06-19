# From the example at https://keras.io/examples/generative/vae/
#

import tensorflow as tf
import keras
import numpy as np
from tensorflow.keras import layers
from plotly import subplots
import plotly.express as px
from plotly.subplots import make_subplots

# Sampling Layer
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Latent space dimensionality
ldim = 2
# Encoder (not a sequential model)
encoder_inputs = tf.keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, (3, 3), activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, (3, 3), activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
# Vector of means
z_mean = layers.Dense(ldim, name="z_mean")(x)
# Vector of log variances
z_log_var = layers.Dense(ldim, name="z_log_var")(x)
# Point in latent space
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z])

# Decoder
latent_inputs = keras.Input(shape=(ldim,))
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
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

    # Training step for one batch
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.vae_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.vae_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }

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

# Encoder-decoder use
encoded_imgs = encoder.predict(x_test)
decoded_imgs = np.squeeze(decoder.predict(encoded_imgs[2]))
# Plot the original and decoded images
fig = make_subplots(rows=2, cols=10)
for i in range(10):
    fig.add_trace(px.imshow(x_test[i], binary_string=True).data[0], row=1, col=i + 1)
    fig.add_trace(px.imshow(decoded_imgs[i], binary_string=True).data[0], row=2, col=i + 1)
fig.show(renderer='browser')
fig.write_image('vae_sample.png', height=400, width=1600)

# Generative use, sample a batch of random numbers
# from a multivariate normal Gaussian
sample = Sampling()([np.zeros((128, ldim)), np.zeros((128, ldim))])
# Decode to images
decoded_imgs = np.squeeze(decoder.predict(sample))
# Plot the generated images
fig = make_subplots(rows=1, cols=10)
for i in range(10):
    fig.add_trace(px.imshow(decoded_imgs[i], binary_string=True).data[0], row=1, col=i + 1)
fig.show(renderer='browser')
fig.write_image('vae_sample_generative.png', height=400, width=1600)

# Create a 2D plot of decoded images across a 2D latent space
figure = np.zeros((28 * 10, 28 * 10))
# linearly spaced coordinates corresponding to the 2D plot
# of digit classes in the latent space
grid_x = np.linspace(-1.0, 1.0, 10)
grid_y = np.linspace(-1.0, 1.0, 10)[::-1]

for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi]])
        x_decoded = vae.decoder.predict(z_sample)
        digit = x_decoded[0].reshape(28, 28)
        figure[
            i * 28 : (i + 1) * 28,
            j * 28 : (j + 1) * 28,
        ] = digit

fig = px.imshow(figure, binary_string=True)
fig.show(renderer='browser')
fig.write_image('vae_sample_generative2.png', height=400, width=1600)

