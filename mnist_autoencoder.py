# This example based on code by Kevin P. Murphy
# "Probabilistic Machine Learning - An Introduction"
#

from tensorflow.keras import layers
from tensorflow.keras import models
import keras
from plotly import subplots
import plotly.express as px
from plotly.subplots import make_subplots

# Convolutional model encoder
encoder = models.Sequential([
    layers.Reshape([28, 28, 1], input_shape=[28, 28]),
    layers.Conv2D(16, (3, 3), padding="same", activation="relu"),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.MaxPool2D((2, 2)),
])

# Convolutional model decoder
decoder = models.Sequential([
    layers.Conv2DTranspose(
            32, (3, 3), strides=2, padding="VALID", activation="relu", input_shape=[3, 3, 64]
        ),
    layers.Conv2DTranspose(16, (3, 3), strides=2, padding="SAME", activation="selu"),
    layers.Conv2DTranspose(1, (3, 3), strides=2, padding="SAME", activation="sigmoid"),
    layers.Reshape([28, 28]),
])

# Complete auto-encoder model
auto_encoder = models.Sequential([
    encoder, 
    decoder
])

# Load the fashion mnist data set
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
# Transform values from ints to floats in [0,1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Compile and train the auto-encoder
auto_encoder.compile(loss="mse")
auto_encoder.fit(x_train, x_train, epochs=5, validation_data=(x_test, x_test))

# Encode and decode some digits
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

fig = make_subplots(rows=2, cols=10)
for i in range(10):
    fig.add_trace(px.imshow(x_test[i], binary_string=True).data[0], row=1, col=i + 1)
    fig.add_trace(px.imshow(decoded_imgs[i], binary_string=True).data[0], row=2, col=i + 1)
fig.show(renderer='browser')
fig.write_image('autoencoder_sample_cnn.png', height=400, width=1600)
