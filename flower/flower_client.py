import collections
import functools
import os
import sys
import time
import numpy as np
import tensorflow as tf
import pandas as pd
import flwr as fl

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from model_helper_flower import ModelHelper

# Select the clients
sample_clients = [457, 461, 228, 153, 445]
if len(sys.argv) > 1:
    index = int(sys.argv[1])
else:
    index = 0

client_id = sample_clients[index]

# read the dataset from Drive
# csv_path = "../cincinnati/cincinatti_zones_" + str(client_id) + ".csv"

# read dataset when on pi
csv_path = "./cincinatti_zones_" + str(client_id) + ".csv"
print("start reading dataset...")
print(os.getcwd())
df = pd.read_csv(csv_path)
print("dataset loaded!")

locations = df.location_id
vocab_size = 100

user_df = df.loc[df.vehicle_id == client_id].copy()
user_df.drop(['vehicle_id'], axis=1, inplace=True)

# List of numerical column names
numerical_column_names = ['clock_sin', 'clock_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 'week_day_sin',
                          'week_day_cos']

mh = ModelHelper(user_df, 17)
mh.set_target_column_name('location_id')
mh.set_vocab_size(vocab_size)
column_names = ['location_id'] + numerical_column_names
mh.set_column_names(column_names)
mh.set_numerical_column_names(numerical_column_names)

# Split the data inti train, val, test.
mh.train_val_test_split()
mh.split_data()

# Create the final datasets for training.
mh.set_batch_size(16)
mh.create_and_batch_datasets(multi_target=False)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units1 = 256
rnn_units2 = 128
print("prep done!")


# Create a model
def create_keras_model():
    N = mh.total_window_length
    batch_size = mh.batch_size
    number_of_places = mh.vocab_size

    # Shortcut to the layers package
    l = tf.keras.layers

    # List of numeric feature columns to pass to the DenseLayer
    numeric_feature_columns = []

    # Handling numerical columns
    for header in numerical_column_names:
        # Append all the numerical columns defined into the list
        numeric_feature_columns.append(feature_column.numeric_column(header, shape=N - 1))

    feature_inputs = {}
    for c_name in numerical_column_names:
        feature_inputs[c_name] = tf.keras.Input((N - 1,), batch_size=batch_size, name=c_name)

    # We cannot use an array of features as always because we have sequences
    # We have to do one by one in order to match the shape
    num_features = []
    for c_name in numerical_column_names:
        f = feature_column.numeric_column(c_name, shape=(N - 1))
        feature = l.DenseFeatures(f)(feature_inputs)
        feature = tf.expand_dims(feature, -1)
        num_features.append(feature)

    # Declare the dictionary for the categories sequence as before
    sequence_input = {
        'location_id': tf.keras.Input((N - 1,), batch_size=batch_size, dtype=tf.dtypes.int32, name='location_id')
        # add batch_size=batch_size in case of stateful GRU
    }

    # Handling the categorical feature sequence using one-hot
    location_one_hot = feature_column.sequence_categorical_column_with_vocabulary_list(
        'location_id', [i for i in range(number_of_places)])

    # one-hot encoding
    location_feature = feature_column.embedding_column(location_one_hot, embedding_dim)

    # With an input sequence we can't use the DenseFeature layer, we need to use the SequenceFeatures
    sequence_features, sequence_length = tf.keras.experimental.SequenceFeatures(location_feature)(sequence_input)

    input_sequence = l.Concatenate(axis=2)([sequence_features] + num_features)

    # Rnn
    recurrent = l.GRU(rnn_units1,
                      batch_size=batch_size,  # in case of stateful
                      return_sequences=True,
                      stateful=True,
                      recurrent_initializer='glorot_uniform')(input_sequence)

    recurrent_2 = l.GRU(rnn_units2,
                        batch_size=batch_size,  # in case of stateful
                        stateful=True,
                        recurrent_initializer='glorot_uniform')(recurrent)

    # Softmax output layer
    # Last layer with an output for each places
    output = layers.Dense(number_of_places, activation='softmax')(recurrent_2)

    # To return the Model, we need to define its inputs and outputs
    # In out case, we need to list all the input layers we have defined
    inputs = list(feature_inputs.values()) + list(sequence_input.values())

    # Return the Model
    return tf.keras.Model(inputs=inputs, outputs=output)


# Get the model and compile it
mh.assign_model(create_keras_model())
mh.compile_model()
mh.set_num_epochs(30)
mh.fit_model()
mh.evaluate_model()


class Client(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = cid

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return mh.model.get_weights()

    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit, config: {config}")
        mh.model.set_weights(parameters)
        N_EPOCHS = 1
        # Fit the model
        mh.model.fit(mh.train_dataset, epochs=mh.num_epochs)
        return mh.model.get_weights(), len(mh.train_dataset), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        mh.model.set_weights(parameters)
        loss, sparse_categorical_accuracy = mh.model.evaluate(mh.val_dataset)
        print(loss, len(mh.val_dataset), {"sparse_categorical_accuracy": sparse_categorical_accuracy})
        return loss, len(mh.val_dataset), {"sparse_categorical_accuracy": sparse_categorical_accuracy}


# Start Flower client
# fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=Client(index)) # local host config for simulation
fl.client.start_numpy_client(server_address="192.168.178.41:8080", client=Client(index))

# duration per epoch on pi 160s - 230s with train data size 700-900
