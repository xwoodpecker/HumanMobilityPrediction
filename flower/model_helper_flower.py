#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import collections
import numpy as np

from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split


class ModelHelper:

    def __init__(self, df, total_window_length):
        self.df = df
        self.sequence_length = total_window_length - 1
        self.total_window_length = total_window_length
        self.target_column_name = df.columns[0]
        self.num_epochs = 20
        self.client_column = 'user_id'
        self.client_column_ids = None

    def reset_df(self, df):
        self.df = df

    def train_val_test_split(self):
        train, self.df_test = train_test_split(self.df, test_size=0.2, shuffle=False)
        self.df_train, self.df_val = train_test_split(train, test_size=0.2, shuffle=False)

    def set_column_names(self, column_names):
        self.column_names = column_names

    def set_numerical_column_names(self, numerical_column_names):
        self.numerical_column_names = numerical_column_names

    def set_target_column_name(self, target_column_name):
        self.target_column_name = target_column_name

    def set_vocab_size(self, vocab_size=None):
        if vocab_size is None:
            self.vocab_size = int(np.max(self.df[self.target_column_name].values) + 1)
        else:
            self.vocab_size = vocab_size

    def set_client_column_name(self, column_name):
        self.client_column = column_name
        self.set_client_column_ids()

    def set_client_column_ids(self, ids=None):
        if ids is None:
            self.client_column_ids = self.df[self.client_column].unique()
        else:
            self.client_column_ids = ids

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_num_epochs(self, num_epochs):
        self.num_epochs = num_epochs

    def set_hyper_parameters(self, num_epochs, batch_size, prefetch_buffer):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.prefetch_buffer = prefetch_buffer

    def get_unique_clients(self):
        if self.client_column_ids is None and self.client_column in self.df:
            return self.df[self.client_column].unique()
        return self.client_column_ids

    # Split the data into chunks of total_window_length
    def split_data(self):
        # Get a list of dataframes of length total_window_length records
        list_train = (
            [self.df_train[i:i + self.total_window_length] for i in
             range(0, self.df_train.shape[0], self.total_window_length)])
        list_val = (
            [self.df_val[i:i + self.total_window_length] for i in
             range(0, self.df_val.shape[0], self.total_window_length)])
        list_test = (
            [self.df_test[i:i + self.total_window_length] for i in
             range(0, self.df_test.shape[0], self.total_window_length)])

        self.list_train = list_train
        self.list_val = list_val
        self.list_test = list_test

    # Split data for FL

    # In[5]:

    def split_data_users(self):
        # dictionary of list of df
        df_dictionary = {}

        for uid in tqdm(self.client_column_ids):
            # Get the records of the user
            user_df_train = self.df_train.loc[self.df_train[self.client_column] == uid].copy()
            user_df_val = self.df_val.loc[self.df_val[self.client_column] == uid].copy()
            user_df_test = self.df_test.loc[self.df_test[self.client_column] == uid].copy()

            # Get a list of dataframes of length N records
            user_list_train = [user_df_train[i:i + self.total_window_length] for i in
                               range(0, user_df_train.shape[0], self.total_window_length)]
            user_list_val = [user_df_val[i:i + self.total_window_length] for i in
                             range(0, user_df_val.shape[0], self.total_window_length)]
            user_list_test = [user_df_test[i:i + self.total_window_length] for i in
                              range(0, user_df_test.shape[0], self.total_window_length)]

            # Save the list of dataframes into a dictionary
            df_dictionary[uid] = {
                'train': user_list_train,
                'val': user_list_val,
                'test': user_list_test
            }

        self.df_dictionary = df_dictionary

    def sliding_window(self, arr):
        """
        Splits an array into a list of subarrays using a sliding window approach.

        Parameters:
            arr: a numpy array
            N: the number of elements in each subarray

        Returns:
            A list of numpy arrays.
        """
        # Get the number of subarrays
        num_subarrays = arr.shape[0] - self.total_window_length + 1

        # Create an empty list to store the subarrays
        subarrays = []

        # Loop through the array and create the subarrays
        for i in range(num_subarrays):
            subarray = arr[i:i + self.total_window_length]
            subarrays.append(subarray)

        return subarrays

    # Split the data into chunks of N
    def split_data_sliding(self):
        return self.sliding_window(self.df_train), self.sliding_window(self.df_val), self.sliding_window(self.df_test)

    # Create TF DataSet for Basic (Non-Federated)
    # Takes a dictionary with train, validation and test sets and the desired set type
    def create_dataset(self, my_list, multi_target=True):
        input_dict = collections.OrderedDict()

        # If the last dataframe of the list is not complete
        if len(my_list[-1]) < self.total_window_length:
            diff = 1
        else:
            diff = 0

        if len(my_list) > 0:
            # Create the dictionary to create a clientData
            for header in self.column_names:
                input_dict[header] = [my_list[i][header].values[:-1] for i in range(0, len(my_list) - diff)]

        if multi_target is True:
            dataset = tf.data.Dataset.from_tensor_slices((input_dict, np.array(
                [my_list[i][self.column_names[0]].values[1:] for i in range(0, len(my_list) - diff)])))
        else:
            dataset = tf.data.Dataset.from_tensor_slices((input_dict, np.array(
                [my_list[i][self.column_names[0]].values[-1] for i in range(0, len(my_list) - diff)])))

        return dataset

    def create_and_batch_datasets(self, multi_target=True):
        self.train_dataset = self.create_dataset(self.list_train, multi_target).batch(self.batch_size,
                                                                                      drop_remainder=True)
        self.val_dataset = self.create_dataset(self.list_val, multi_target).batch(self.batch_size, drop_remainder=True)
        self.test_dataset = self.create_dataset(self.list_test, multi_target).batch(self.batch_size,
                                                                                    drop_remainder=True)

    # Create Federated Learning Client Dictionaries

    # In[10]:

    # Takes a dictionary with train, validation and test sets and the desired set type
    # for users for federated training
    def create_clients_dict(self, set_type):
        dataset_dict = {}

        for uid in tqdm(self.client_column_ids):

            c_data = collections.OrderedDict()
            values = self.df_dictionary[uid][set_type]

            # If the last dataframe of the list is not complete #
            if len(values[-1]) < self.windowlength:
                diff = 1
            else:
                diff = 0

            if len(values) > 0:
                # Create the dictionary to create a clientData
                for header in self.column_names:
                    c_data[header] = [values[i][header].values for i in range(0, len(values) - diff)]
                dataset_dict[uid] = c_data

        return dataset_dict

    # In[11]:

    def generate_client_dicts(self):
        self.clients_train_dict = self.create_clients_dict('train')
        self.clients_val_dict = self.create_clients_dict('val')
        self.clients_test_dict = self.create_clients_dict('test')

    # FL Preprocessing routine

    # In[12]:

    # preprocess dataset to tf format
    def preprocess(self, dataset, N):
        def batch_format_fn(element):
            x = collections.OrderedDict()

            for name in self.column_names:
                x[name] = tf.reshape(element[name][:, :-1], [-1, N - 1])

            y = tf.reshape(element[self.column_names[0]][:, 1:], [-1, N - 1])

            return collections.OrderedDict(x=x, y=y)

        return dataset.repeat(self.num_epochs).batch(self.batch_size, drop_remainder=True).map(
            batch_format_fn).prefetch(
            self.prefetch_buffer)


    def make_federated_data(self, client_data, client_ids):
        return [
            self.preprocess(client_data.create_tf_dataset_for_client(x), self.total_window_length)
            for x in tqdm(client_ids)
        ]

    # In[15]:

    def generate_federated_data(self, sample_clients):
        # Federate the clients datasets
        self.federated_train_data = self.make_federated_data(self.client_train_data, sample_clients)
        self.federated_val_data = self.make_federated_data(self.client_val_data, sample_clients)
        self.federated_test_data = self.make_federated_data(self.client_test_data, sample_clients)

    # NYC Central (Non-Federated) Model Functions

    # In[16]:

    # function to remove duplicates
    def create_sequence(locations):
        # Flatten the list of places
        sequence = np.reshape(locations.values, [-1])

        # Create a temporary array of the same lenght of the sequece of locations
        copy = np.zeros(sequence.shape[0], dtype=np.int32)

        # Copy the sequence of location in the copy array but shifted right by 1 position
        # The last location does not need to be copied, it can't be a duplicate
        copy[1:] = sequence[:sequence.shape[0] - 1]

        # Where we get 0 it can be a possible duplicated
        duplicated = sequence - copy

        # indices where the subtraction gives 0
        idx = np.where(duplicated == 0)[0]

        # Find where the position of the zeros are even
        even = idx % 2 == 0

        # List the indices where the position is even and the subtraction gave 0
        to_drop = idx[even]

        # Remove the duplicates
        clean_sequence = np.delete(sequence, to_drop)
        return clean_sequence, to_drop

    def create_batch_dataset(self):
        if len(self.list_train[-1]) < self.total_window_length:
            diff_train = 1
        else:
            diff_train = 0

        if len(self.list_val[-1]) < self.total_window_length:
            diff_val = 1
        else:
            diff_val = 0

        if len(self.list_test[-1]) < self.total_window_length:
            diff_test = 1
        else:
            diff_test = 0

        # Define the input features of the  dataset
        train_input_dict = {
            'start_place': np.array(
                [self.list_train[i]['location_id'].values[:-1] for i in range(0, len(self.list_train) - diff_train)]),
            'start_hour_sin': np.array(
                [self.list_train[i]['hour_sin'].values[:-1] for i in range(0, len(self.list_train) - diff_train)]),
            'start_hour_cos': np.array(
                [self.list_train[i]['hour_cos'].values[:-1] for i in range(0, len(self.list_train) - diff_train)]),
            'weekend': np.array(
                [self.list_train[i]['weekend'].values[:-1] for i in range(0, len(self.list_train) - diff_train)]),
            'week_day_sin': np.array(
                [self.list_train[i]['week_day_sin'].values[:-1] for i in range(0, len(self.list_train) - diff_train)]),
            'week_day_cos': np.array(
                [self.list_train[i]['week_day_cos'].values[:-1] for i in range(0, len(self.list_train) - diff_train)]),
            'end_hour_sin': np.array(
                [self.list_train[i]['hour_sin'].values[-1] for i in range(0, len(self.list_train) - diff_train)]),
            'end_hour_cos': np.array(
                [self.list_train[i]['hour_cos'].values[-1] for i in range(0, len(self.list_train) - diff_train)]),
            'end_weekend': np.array(
                [self.list_train[i]['weekend'].values[-1] for i in range(0, len(self.list_train) - diff_train)]),
            'end_week_day_sin': np.array(
                [self.list_train[i]['week_day_sin'].values[-1] for i in range(0, len(self.list_train) - diff_train)]),
            'end_week_day_cos': np.array(
                [self.list_train[i]['week_day_cos'].values[-1] for i in range(0, len(self.list_train) - diff_train)]),
        }

        # Define the input features of the  dataset
        val_input_dict = {
            'start_place': np.array(
                [self.list_val[i]['location_id'].values[:-1] for i in range(0, len(self.list_val) - diff_val)]),
            'start_hour_sin': np.array(
                [self.list_val[i]['hour_sin'].values[:-1] for i in range(0, len(self.list_val) - diff_val)]),
            'start_hour_cos': np.array(
                [self.list_val[i]['hour_cos'].values[:-1] for i in range(0, len(self.list_val) - diff_val)]),
            'weekend': np.array(
                [self.list_val[i]['weekend'].values[:-1] for i in range(0, len(self.list_val) - diff_val)]),
            'week_day_sin': np.array(
                [self.list_val[i]['week_day_sin'].values[:-1] for i in range(0, len(self.list_val) - diff_val)]),
            'week_day_cos': np.array(
                [self.list_val[i]['week_day_cos'].values[:-1] for i in range(0, len(self.list_val) - diff_val)]),
            'end_hour_sin': np.array(
                [self.list_val[i]['hour_sin'].values[-1] for i in range(0, len(self.list_val) - diff_val)]),
            'end_hour_cos': np.array(
                [self.list_val[i]['hour_cos'].values[-1] for i in range(0, len(self.list_val) - diff_val)]),
            'end_weekend': np.array(
                [self.list_val[i]['weekend'].values[-1] for i in range(0, len(self.list_val) - diff_val)]),
            'end_week_day_sin': np.array(
                [self.list_val[i]['week_day_sin'].values[-1] for i in range(0, len(self.list_val) - diff_val)]),
            'end_week_day_cos': np.array(
                [self.list_val[i]['week_day_cos'].values[-1] for i in range(0, len(self.list_val) - diff_val)]),
        }

        # Define the input features of the  dataset
        test_input_dict = {
            'start_place': np.array(
                [self.list_test[i]['location_id'].values[:-1] for i in range(0, len(self.list_test) - diff_test)]),
            'start_hour_sin': np.array(
                [self.list_test[i]['hour_sin'].values[:-1] for i in range(0, len(self.list_test) - diff_test)]),
            'start_hour_cos': np.array(
                [self.list_test[i]['hour_cos'].values[:-1] for i in range(0, len(self.list_test) - diff_test)]),
            'weekend': np.array(
                [self.list_test[i]['weekend'].values[:-1] for i in range(0, len(self.list_test) - diff_test)]),
            'week_day_sin': np.array(
                [self.list_test[i]['week_day_sin'].values[:-1] for i in range(0, len(self.list_test) - diff_test)]),
            'week_day_cos': np.array(
                [self.list_test[i]['week_day_cos'].values[:-1] for i in range(0, len(self.list_test) - diff_test)]),
            'end_hour_sin': np.array(
                [self.list_test[i]['hour_sin'].values[-1] for i in range(0, len(self.list_test) - diff_test)]),
            'end_hour_cos': np.array(
                [self.list_test[i]['hour_cos'].values[-1] for i in range(0, len(self.list_test) - diff_test)]),
            'end_weekend': np.array(
                [self.list_test[i]['weekend'].values[-1] for i in range(0, len(self.list_test) - diff_test)]),
            'end_week_day_sin': np.array(
                [self.list_test[i]['week_day_sin'].values[-1] for i in range(0, len(self.list_test) - diff_test)]),
            'end_week_day_cos': np.array(
                [self.list_test[i]['week_day_cos'].values[-1] for i in range(0, len(self.list_test) - diff_test)]),
        }

        # Create training examples / targets, we are going to predict the next location
        trips_dataset_train = tf.data.Dataset.from_tensor_slices((train_input_dict, np.array(
            [self.list_train[i]['location_id'].values[-1] for i in range(0, len(self.list_train) - diff_train)])))
        trips_dataset_val = tf.data.Dataset.from_tensor_slices((val_input_dict, np.array(
            [self.list_val[i]['location_id'].values[-1] for i in range(0, len(self.list_val) - diff_val)])))
        trips_dataset_test = tf.data.Dataset.from_tensor_slices((test_input_dict, np.array(
            [self.list_test[i]['location_id'].values[-1] for i in range(0, len(self.list_test) - diff_test)])))

        # Buffer size to shuffle the dataset
        # (TF data is designed to work with possibly infinite sequences,
        # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
        # it maintains a buffer in which it shuffles elements).
        # BUFFER_SIZE = 10000

        # Create the dataset by creating batches
        # Uncomment the shuffle function in case we want to shuffle the sequences
        self.train_dataset = trips_dataset_train.batch(self.batch_size, drop_remainder=True)  # .shuffle(BUFFER_SIZE)
        self.val_dataset = trips_dataset_val.batch(self.batch_size, drop_remainder=True)  # .shuffle(BUFFER_SIZE)
        self.test_dataset = trips_dataset_test.batch(self.batch_size, drop_remainder=True)  # .shuffle(BUFFER_SIZE)

        return self.train_dataset, self.val_dataset, self.test_dataset

    def create_users_locations(self):
        users_locations = []

        # For each user
        df = self.df
        for c_id in tqdm(self.get_unique_clients()):
            # Call the function
            self.reset_df(df.loc[df[self.client_column] == c_id].copy())
            locations_sequence, pos, loc = self.df_to_location_sequence()
            # Add the sequence df of the user to the list
            users_locations.append(locations_sequence)

        self.users_locations = users_locations
        return users_locations

    def split_concat_user_df(self):
        users_locations_train = []
        users_locations_val = []
        users_locations_test = []

        for user_df in self.users_locations:
            # Split in train, test and validation
            train, test = train_test_split(user_df, test_size=0.2, shuffle=False)
            train, val = train_test_split(train, test_size=0.2, shuffle=False)

            # Append the sets
            users_locations_train.append(train)
            users_locations_val.append(val)
            users_locations_test.append(test)

        df_train = pd.concat(users_locations_train)
        df_train.drop(['index', 'day', 'month'], axis=1, inplace=True)
        df_train['location_id'] = pd.to_numeric(df_train['location_id'], downcast='integer')
        self.df_train = df_train

        # Merge back the dataframes
        df_val = pd.concat(users_locations_val)
        df_val.drop(['index', 'day', 'month'], axis=1, inplace=True)
        df_val['location_id'] = pd.to_numeric(df_val['location_id'], downcast='integer')
        self.df_val = df_train

        # Merge back the dataframes
        df_test = pd.concat(users_locations_test)
        df_test.drop(['index', 'day', 'month'], axis=1, inplace=True)
        df_test['location_id'] = pd.to_numeric(df_test['location_id'], downcast='integer')
        self.df_test = df_train

    def basic_split_df(self):
        self.df_train, self.df_test = train_test_split(self.df, test_size=0.2, shuffle=False)
        self.df_train, self.df_val = train_test_split(self.df_train, test_size=0.2, shuffle=False)

    def drop_all_but_target(self):
        columns = list(set(self.df_train.columns) - set([self.target_column_name]))

        self.df_train.drop(columns, axis=1, inplace=True)
        self.df_val.drop(columns, axis=1, inplace=True)
        self.df_test.drop(columns, axis=1, inplace=True)

        self.df_train['location_id'] = pd.to_numeric(self.df_train['location_id'], downcast='integer')
        self.df_val['location_id'] = pd.to_numeric(self.df_val['location_id'], downcast='integer')
        self.df_test['location_id'] = pd.to_numeric(self.df_test['location_id'], downcast='integer')

    def set_window_generator(self, label_columns=None):
        window_generator = WindowGenerator(self.sequence_length, 1, 1, self.df_train, self.df_val, self.df_test,
                                           label_columns)
        window_generator.batch_size = self.batch_size
        self.window_generator = window_generator

    def make_windowed_dataset(self, sequence_stride=1):
        self.train_dataset = self.window_generator.make_dataset(self.window_generator.train_df, sequence_stride)
        self.val_dataset = self.window_generator.make_dataset(self.window_generator.val_df, sequence_stride)
        self.test_dataset = self.window_generator.make_dataset(self.window_generator.test_df, sequence_stride)

    def assign_model(self, model):
        self.model = model

    # In[29]:

    adam = tf.keras.optimizers.Adam(learning_rate=0.001)

    def compile_model(self, optimizer_type=tf.keras.optimizers.Adam, learning_rate=0.001):
        optimizer = optimizer_type(learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics=tf.keras.metrics.SparseCategoricalAccuracy())

    # In[30]:

    def fit_model(self, with_early_stopping=True):
        callbacks = []
        if with_early_stopping is True:
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=3,
                                                              restore_best_weights=True, verbose=1))

        self.model.fit(self.train_dataset, validation_data=self.val_dataset, epochs=self.num_epochs,
                       callbacks=callbacks)

    def evaluate_model(self):
        self.model.evaluate(self.test_dataset)

    def print_test_prediction_info(self):
        def print_info(arr, name):
            print(name)
            print('Shape : ', arr.shape)
            print('Example [0] : ', arr[0])
            if len(arr.shape) > 2:
                print('2nd axis example : ', arr[0][:][0])

        logits = self.model.predict(self.test_dataset)
        print_info(logits, 'logits')

        predictions = tf.nn.softmax(logits, axis=1)
        print_info(predictions, 'predictions')

        predicted_classes = np.argmax(predictions, 1)
        print_info(predicted_classes, 'predicted_classes')

        actual_values = []
        for x, y in self.test_dataset.unbatch():
            actual_values.append(y.numpy())

        actual_values = np.array(actual_values)
        print_info(actual_values, 'actual_values')

        diff = actual_values - predicted_classes
        print_info(diff, 'diff')

        for i in range(0, 20):
            print('Prediction #', i)
            print('Actual values: ', actual_values[i])
            print('Predicted values: ', predicted_classes[i])

        wrong = np.count_nonzero(diff)
        size = diff.shape[0]
        correct = size - wrong
        acc = correct / size
        print('# correct Predictions : ', correct)
        print('# wrong Predictions : ', wrong)
        print('accuracy: ', acc)


def df_to_dataset(dataframe, shuffle=False, batch_size=32):
    labels = dataframe.pop('cat_id')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


ModelHelper.df_to_dataset = df_to_dataset


# Window Generator Approach
# 
# modified and manually imported "timeseries_dataset_from_array"

# In[19]:


# IMPORTANT CHANGE : "drop_remainder" SET TO True SO THAT THE DATASET IS ALWAYS THE RIGHT SIZE
# (THIS GUARANTEES HAVING NO INCOMPLETE SHAPES IN LAST THE BATCH)
# dataset = dataset.batch(batch_size, drop_remainder=True)

def timeseries_dataset_from_array(
        data,
        targets,
        sequence_length,
        sequence_stride=1,
        sampling_rate=1,
        batch_size=128,
        shuffle=False,
        seed=None,
        start_index=None,
        end_index=None,
):
    if start_index:
        if start_index < 0:
            raise ValueError(
                "`start_index` must be 0 or greater. Received: "
                f"start_index={start_index}"
            )
        if start_index >= len(data):
            raise ValueError(
                "`start_index` must be lower than the length of the "
                f"data. Received: start_index={start_index}, for data "
                f"of length {len(data)}"
            )
    if end_index:
        if start_index and end_index <= start_index:
            raise ValueError(
                "`end_index` must be higher than `start_index`. "
                f"Received: start_index={start_index}, and "
                f"end_index={end_index} "
            )
        if end_index >= len(data):
            raise ValueError(
                "`end_index` must be lower than the length of the "
                f"data. Received: end_index={end_index}, for data of "
                f"length {len(data)}"
            )
        if end_index <= 0:
            raise ValueError(
                "`end_index` must be higher than 0. "
                f"Received: end_index={end_index}"
            )

    # Validate strides
    if sampling_rate <= 0:
        raise ValueError(
            "`sampling_rate` must be higher than 0. Received: "
            f"sampling_rate={sampling_rate}"
        )
    if sampling_rate >= len(data):
        raise ValueError(
            "`sampling_rate` must be lower than the length of the "
            f"data. Received: sampling_rate={sampling_rate}, for data "
            f"of length {len(data)}"
        )
    if sequence_stride <= 0:
        raise ValueError(
            "`sequence_stride` must be higher than 0. Received: "
            f"sequence_stride={sequence_stride}"
        )
    if sequence_stride >= len(data):
        raise ValueError(
            "`sequence_stride` must be lower than the length of the "
            f"data. Received: sequence_stride={sequence_stride}, for "
            f"data of length {len(data)}"
        )

    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = len(data)

    # Determine the lowest dtype to store start positions (to lower memory
    # usage).
    num_seqs = end_index - start_index - (sequence_length * sampling_rate) + 1
    if targets is not None:
        num_seqs = min(num_seqs, len(targets))
    if num_seqs < 2147483647:
        index_dtype = "int32"
    else:
        index_dtype = "int64"

    # Generate start positions
    start_positions = np.arange(0, num_seqs, sequence_stride, dtype=index_dtype)
    if shuffle:
        if seed is None:
            seed = np.random.randint(1e6)
        rng = np.random.RandomState(seed)
        rng.shuffle(start_positions)

    sequence_length = tf.cast(sequence_length, dtype=index_dtype)
    sampling_rate = tf.cast(sampling_rate, dtype=index_dtype)

    positions_ds = tf.data.Dataset.from_tensors(start_positions).repeat()

    # For each initial window position, generates indices of the window elements
    indices = tf.data.Dataset.zip(
        (tf.data.Dataset.range(len(start_positions)), positions_ds)
    ).map(
        lambda i, positions: tf.range(
            positions[i],
            positions[i] + sequence_length * sampling_rate,
            sampling_rate,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    dataset = sequences_from_indices(data, indices, start_index, end_index)
    if targets is not None:
        indices = tf.data.Dataset.zip(
            (tf.data.Dataset.range(len(start_positions)), positions_ds)
        ).map(
            lambda i, positions: positions[i],
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        target_ds = sequences_from_indices(
            targets, indices, start_index, end_index
        )
        dataset = tf.data.Dataset.zip((dataset, target_ds))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    if batch_size is not None:
        if shuffle:
            # Shuffle locally at each iteration
            dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
        dataset = dataset.batch(batch_size, drop_remainder=True)
    else:
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1024, seed=seed)
    return dataset


# In[20]:


def sequences_from_indices(array, indices_ds, start_index, end_index):
    dataset = tf.data.Dataset.from_tensors(array[start_index:end_index])
    dataset = tf.data.Dataset.zip((dataset.repeat(), indices_ds)).map(
        lambda steps, inds: tf.gather(steps, inds),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return dataset


class WindowGenerator:

    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df, label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])


# In[24]:


def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels


WindowGenerator.split_window = split_window


# In[25]:


def make_dataset(self, data, sequence_stride=1):
    data = np.array(data, dtype=np.float32)
    ds = timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=sequence_stride,
        shuffle=False,
        batch_size=self.batch_size, )

    ds = ds.map(self.split_window)

    return ds


WindowGenerator.make_dataset = make_dataset


# In[26]:


# legacy function to fix train, val and test datasets

# In[ ]:


def fix_length_dfs(self):
    def fix_length_df(df, batch_size):
        if df.shape[0] % batch_size == 0:
            return df

        idx = -1 * (df.shape[0] % batch_size)
        print(idx)
        return df[:idx]

    fix_length_df(self.df_train, self.batch_size)
    fix_length_df(self.df_val, self.batch_size)
    fix_length_df(self.df_test, self.batch_size)
