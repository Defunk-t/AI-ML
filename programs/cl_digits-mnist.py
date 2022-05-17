import tensorflow as tf

# Load MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Build a Sequential model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

#
predictions = model(x_train[:1]).numpy()
# And convert to probability
tf.nn.softmax(predictions).numpy()

# Set up loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()

# Compile, train & evaluate model
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)

# Probability model
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])(x_test[:5])
