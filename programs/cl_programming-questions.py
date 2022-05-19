import os
import re
import math
import string
import sys
import random
import tensorflow as tf
import matplotlib.pyplot as pyplot

#
# VARIABLES
#

BATCH_SIZE = 32
SEED = random.randint(0, 99999)
VALIDATION_SPLIT = 0.2
MAX_TOKENS = 10000
EPOCHS = 50
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 2
OPTIMISER = 'adam'

MODEL_FILE = "../models/stack-overflow/model"

# Parse arguments
for i in range(1, len(sys.argv)):
    if sys.argv[i] == '--seed':
        i = i + 1
        SEED = int(sys.argv[i])
    elif sys.argv[i] == '--epochs':
        i = i + 1
        EPOCHS = int(sys.argv[i])
    elif sys.argv[i] == '--batch-size':
        i = i + 1
        BATCH_SIZE = int(sys.argv[i])
    elif sys.argv[i] == '--validation-split':
        i = i + 1
        VALIDATION_SPLIT = float(sys.argv[i])
    elif sys.argv[i] == '--max-tokens':
        i = i + 1
        MAX_TOKENS = int(sys.argv[i])
    elif sys.argv[i] == '--no-early-stopping':
        EARLY_STOPPING = False
    elif sys.argv[i] == '--early-stopping-patience':
        i = i + 1
        EARLY_STOPPING_PATIENCE = int(sys.argv[i])
    elif sys.argv[i] == '--optimiser':
        i = i + 1
        OPTIMISER = sys.argv[i]

#
# IMPORT DATASET
#

# Download
dataset = tf.keras.utils.get_file(
    origin="https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz",
    extract=True,
    cache_dir='../',
    cache_subdir='datasets/stack-overflow'
)

# Set paths
dataset_dir = os.path.dirname(dataset)
train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'test')

# The 'text_dataset_from_directory' function will import
# text files that are organised into folders by class

training_data = tf.keras.utils.text_dataset_from_directory(
    train_dir,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    subset='training',
    seed=SEED
)

validation_data = tf.keras.utils.text_dataset_from_directory(
    train_dir,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    subset='validation',
    seed=SEED
)

test_data = tf.keras.utils.text_dataset_from_directory(
    test_dir,
    batch_size=BATCH_SIZE
)

#
# PRE-PROCESS DATA
#


# Each piece of data will be pre-processed by this function.
# It will standardise the data - converting it to lowercase, removing
# punctuation, and replacing any HTML line breaks with a space character.
def standardise(data):
    return tf.strings.regex_replace(
        tf.strings.regex_replace(
            tf.strings.lower(data),
            "<br />", " "
        ),
        "[%s]" % re.escape(string.punctuation), ""
    )


# The data is split into 'tokens' (individual words, delimited by whitespace)
#   unique tokens are given a unique integer in an index
#   the mapped integer is used in the first layer of the model
vectorised_layer = tf.keras.layers.TextVectorization(
    standardize=standardise,
    max_tokens=MAX_TOKENS,
    output_mode='int',
    output_sequence_length=250
)

vectorised_layer.adapt(training_data.map(lambda x, y: x))


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorised_layer(text), label


# Map vectorised data prior to training
training_data_v = training_data.map(vectorize_text)
validation_data_v = validation_data.map(vectorize_text)
test_data_v = test_data.map(vectorize_text)

# Prefetch it in the cache
training_data_v = training_data_v.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
validation_data_v = validation_data_v.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_data_v = test_data_v.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

#
# MODEL SETUP
#

# Build a sequential model
model = tf.keras.Sequential([

    # Looks up embedding vector for each word-index
    tf.keras.layers.Embedding(MAX_TOKENS + 1, 16),

    # Returns a fixed length output vector
    # to handle varying input length (not all
    # reviews are the same number of chars).
    tf.keras.layers.GlobalAveragePooling1D(),

    # 4 output nodes (one per class)
    tf.keras.layers.Dense(4)
])

# model.summary()  # prints a summary of the model

# Compile with the chosen loss function and optimiser algo
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy']
)

#
# MODEL TRAINING
#

callbacks = []

# Reduce over-training by monitoring for increase to val_loss
if EARLY_STOPPING:
    callbacks.append(tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE
    ))

# # Save the model as it is being trained
# callbacks.append(tf.keras.callbacks.ModelCheckpoint(
#     filepath=MODEL_FILE,
#     save_weights_only=True,
#     verbose=1
# ))

# Train or 'fit' the model
model_history = model.fit(
    training_data_v,
    validation_data=validation_data_v,
    epochs=EPOCHS,
    callbacks=callbacks
)

#
# EVALUATION
#

loss, accuracy = model.evaluate(test_data_v)

print("Loss: ", math.floor(loss * 1000) / 10, "%")
print("Accuracy: ", math.floor(accuracy * 1000) / 10, "%")

acc = model_history.history['accuracy']
val_acc = model_history.history['val_accuracy']
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(1, len(acc) + 1)

# Plot and show loss across epochs
pyplot.plot(epochs, loss, 'bo', label="Training loss")
pyplot.plot(epochs, val_loss, 'b', label="Validation loss")
pyplot.title("Training and validation loss")
pyplot.xlabel("Epochs")
pyplot.ylabel("Loss")
pyplot.legend()
pyplot.show()

# Plot and show accuracy across epochs
pyplot.plot(epochs, acc, 'bo', label="Training acc")
pyplot.plot(epochs, val_acc, 'b', label="Validation acc")
pyplot.title("Training and validation accuracy")
pyplot.xlabel("Epochs")
pyplot.ylabel("Accuracy")
pyplot.legend(loc='lower right')
pyplot.show()

#
# EXPORT MODEL
#

# Create another model with the pre-processing "built in"
print("Creating and testing export model...")

final_model = tf.keras.Sequential([
    vectorised_layer,
    model,
    tf.keras.layers.Activation('sigmoid')
])

final_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=OPTIMISER,
    metrics=['accuracy']
)

final_model.save_weights(MODEL_FILE)

# Test it with raw unprocessed data
loss, accuracy = final_model.evaluate(test_data)

print("Accuracy: ", math.floor(accuracy * 1000) / 10, "%")

# Test it with new data
print(final_model.predict([
    # Include an array of strings to test each in the model
    "tries to control an arduino servo motor with blank and I have to send a number with added to the end but I keep popping up a message from TypeError: unsupported operand type(s) for +: 'int' and 'bytes' import serial, time with serial.Serial('COM4', 9600) as ser: f = ser.readline() print(f) x = 400 y = 360 x = int(180 - x // 5) y = int(x // 4) ser.write(x+.encode()) time.sleep(1) ser.write(y+" ".encode()) g = ser.readline() print(g) ser.close()"
    # The above is a Python question
]))
