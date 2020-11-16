
import os
import random # to shuffle

import matplotlib.pyplot as plt # plotting
import pandas as pd # plotting
import seaborn as sn # plotting
from torch.utils import data # for the dataloader


class Dataset(data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        # Return the size of the dataset
        return len(self.targets)

    def __getitem__(self, index):
        # Retrieve inputs and targets at the given index
        X = self.inputs[index]
        y = self.targets[index]

        return X, y


def create_datasets(sequences, dataset_class, p_train=0.8, p_val=0.1, p_test=0.1):

    # sensors
    sensors = [i for i in range(len(set(sequences.sensor_id)))]

    # shuffle the data by sensor
    random.shuffle(sensors)

    # Define partition sizes
    num_train = int(len(sensors)*p_train)
    num_val = int(len(sensors)*p_val)
    num_test = int(len(sensors)*p_test)
    

    # Split sensor indices into partitions
    sensors_train = sensors[:num_train]
    sensors_val = sensors[num_train:num_train+num_val]
    sensors_test = sensors[-num_test:]

    # capture dataframes for the indiced sensors
    sequences_train  = sequences[sequences['sensor_id'].isin(sensors_train)]
    sequences_val  = sequences[sequences['sensor_id'].isin(sensors_val)]
    sequences_test  = sequences[sequences['sensor_id'].isin(sensors_test)]

    def get_inputs_targets_from_sequences(sequences):
        # Define empty lists

        inputs, targets = [], []
        
        # for each sensor
        for (sensor, sequence) in sequences.groupby('sensor_id'):
            inputs.append(sequence['moisture'][:-1].tolist())
            targets.append(sequence['moisture'][1:].tolist())
            
        return inputs, targets

    # Get inputs and targets for each partition
    inputs_train, targets_train = get_inputs_targets_from_sequences(sequences_train)
    inputs_val, targets_val = get_inputs_targets_from_sequences(sequences_val)
    inputs_test, targets_test = get_inputs_targets_from_sequences(sequences_test)

    # # Create datasets
    training_set = dataset_class(inputs_train, targets_train)
    validation_set = dataset_class(inputs_val, targets_val)
    test_set = dataset_class(inputs_test, targets_test)

    return training_set, validation_set, test_set
    
# Importing data
sequences = pd.read_csv(os.getcwd()+'/Data/sensor1.csv')

# capturing the relevant columns
sequences = sequences[['sensor_id','moisture']]

# creating the datasets
training_set, validation_set, test_set = create_datasets(sequences, Dataset)
