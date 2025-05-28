import keras
from keras import layers
import tensorflow_datasets as tfds
from plotly import subplots
import plotly.express as px

# Load a Tensorflow example image data set
train_ds, test_ds = tfds.load(
    "cats_vs_dogs",
    split=["train[:75%]", "train[75%:100%]"],
    as_supervised=True)

# Create Resizing layer
resize_fn = keras.layers.Resizing(150, 150)
# Apply resizing layer to data sets
train_ds = train_ds.map(lambda x, y: (resize_fn(x), y))
test_ds = test_ds.map(lambda x, y: (resize_fn(x), y))

# Create Scaling layer
scale_fn = keras.layers.Rescaling(scale=1.0/255)
# Apply scaling layer to data sets
train_ds = train_ds.map(lambda x, y: (scale_fn(x), y))
test_ds = test_ds.map(lambda x, y: (scale_fn(x), y))

# Show some example images
fig = subplots.make_subplots(rows=5, cols=5)
train_iterator = iter(train_ds)
for i in range(25):
    image, label = next(train_iterator)
    fig.add_trace(px.imshow(image.numpy()).data[0], row=i // 5 + 1, col=i % 5 + 1)
fig.show(renderer='browser')
fig.write_image('catsdogs_sample.png', height=800, width=800)

# Define a sequential model of random transformation layers
augmentation = keras.models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomTranslation(.1, .1, fill_mode="reflect"),
    layers.RandomRotation(0.2, fill_mode="reflect"),
    layers.RandomZoom(.2, .2, fill_mode="reflect")
])
# Apply augmentation model to training data
train_ds = train_ds.map(lambda x, y: (augmentation(x), y))

# Show the transformed sample images
fig = subplots.make_subplots(rows=5, cols=5)
train_iterator = iter(train_ds)
for i in range(25):
    image, label = next(train_iterator)
    fig.add_trace(px.imshow(image.numpy()).data[0], row=i // 5 + 1, col=i % 5 + 1)
fig.show(renderer='browser')
fig.write_image('catsdogs_sample_transformed.png', height=800, width=800)

base_model = keras.applications.Xception(
    weights="imagenet",
    input_shape=(150, 150, 3),
    include_top=False)
base_model.trainable = False

# Create new input model
inputs = keras.Input(shape=(150, 150, 3))
# Pre-trained Xception weights requires inputs in [-1., +1.]
scale_layer = keras.layers.Rescaling(scale=2., offset=-1)
x = scale_layer(inputs)
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(1)(x)

# Complete model has inputs and outputs
model = keras.Model(inputs, outputs)
# Not all parameters are trainable
model.summary(show_trainable=True)
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)
model.fit(train_ds.batch(64), epochs=2,
          validation_data=test_ds.batch(64))

# Make all parameters trainable
base_model.trainable = True
# Now all parameters are trainable
model.summary(show_trainable=True)

# Very small learning rate
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)
model.fit(train_ds.batch(64), epochs=2,
          validation_data=test_ds.batch(64))
