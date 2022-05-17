# Trains a TensorFlow model using the Fashion MNIST dataset

import math
import numpy
import tensorflow as tf
import matplotlib.pyplot as pyplot

# Import the Fashion MNIST dataset
#   Arrays of images and labels - some
#   for training and some for testing
(train_images, train_labels), (test_images, test_labels) \
    = tf.keras.datasets.fashion_mnist.load_data()

# Each label is an integer (0-9) corresponding to one of these classes
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Shows the shape of the image data
# Each image is an array (28x28) of values (0-255)
print("Training images shape: ", train_images.shape)
print("Testing  images shape: ", test_images.shape)

# Also shows that there is an equal number of labels
# 60,000 for training and 10,000 for testing
print(len(train_labels), " training labels")
print(len(test_labels), " testing labels")

# matplotlib.pyplot can be used to show how pixel values are represented
#
# pyplot.figure()
# pyplot.imshow(train_images[0])
# pyplot.colorbar()
# pyplot.grid(False)
# pyplot.show()

# Using pyplot to show some images with their class name
#
# pyplot.figure(figsize=(10, 10))
# for i in range(25):
#     pyplot.subplot(5, 5, i + 1)
#     pyplot.xticks([])
#     pyplot.yticks([])
#     pyplot.grid(False)
#     pyplot.imshow(train_images[i], cmap=pyplot.cm.binary)
#     pyplot.xlabel(class_names[train_labels[i]])
# pyplot.show()

# Scale to floating point values (0 and 1)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Construct a sequential model
model = tf.keras.Sequential([
    # The 2D images need to be flattened to 1D
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # A fully-connected layer with n nodes
    tf.keras.layers.Dense(128, activation='relu'),
    # Output layer, 10 nodes corresponding to class name
    # coming back with a probability score
    tf.keras.layers.Dense(10)
])

# Compile the model with the loss function
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10)

# Evaluate the accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\nAccuracy:", math.floor(test_acc * 1000) / 10, "%")

# Get the probability model from the trained model
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

# Get predictions for the test images
predictions = probability_model.predict(test_images)

# Plot predictions vs. accuracy for first 15 in test set
rows, cols = 5, 3
pyplot.figure(figsize=(2 * 2 * cols, 2 * rows))
for i in range(rows * cols):

    true_label, img = test_labels[i], test_images[i]
    predicted_label = numpy.argmax(predictions[i])

    # Plot the image
    pyplot.subplot(rows, 2 * cols, 2 * i + 1)
    pyplot.grid(False)
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.imshow(img, cmap=pyplot.cm.binary)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    pyplot.xlabel("{} {:2.0f}% ({})".format(
        class_names[predicted_label],
        100 * numpy.max(predictions[i]),
        class_names[true_label]),
        color=color
    )

    # Plot the prediction array
    pyplot.subplot(rows, 2 * cols, 2 * i + 2)
    pyplot.grid(False)
    pyplot.xticks(range(10))
    pyplot.yticks([])
    bar = pyplot.bar(range(10), predictions[i], color="#777777")
    pyplot.ylim([0, 1])
    bar[predicted_label].set_color('red')
    bar[true_label].set_color('blue')

pyplot.tight_layout()
pyplot.show()
