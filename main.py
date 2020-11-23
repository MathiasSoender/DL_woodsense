
import os
import random # to shuffle

import matplotlib.pyplot as plt # plotting
import pandas as pd # plotting
import seaborn as sn # plotting
from torch.utils import data # for the dataloader
import torch
import torch.nn as nn

#############################################################################
################################# DATALOADER ################################
#############################################################################

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
sequences = sequences[['sensor_id','moisture']]
training_set, validation_set, test_set = create_datasets(sequences, Dataset)

#############################################################################
############################## ENCODER DECODER ##############################
#############################################################################


class Encoder(nn.Module):
    
    def __init__(self, hidden_dim = 8):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.lstm1 = nn.LSTMCell(1, hidden_dim)
        self.lstm2 = nn.LSTMCell(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, 1)

def forward(self, input, future = 0):

        outputs = []

        # Initialize hidden states
        h_t = torch.zeros(input.size(0), self.hidden_dim, dtype=torch.double)
        c_t = torch.zeros(input.size(0), self.hidden_dim, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), self.hidden_dim, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), self.hidden_dim, dtype=torch.double)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

encoder = Encoder()
encoder.forward(training_set)

#############################################################################
################################# TRAINING ##################################
#############################################################################

# # create an optimizer object
# # Adam optimizer with learning rate 1e-3
# optimizer = optim.Adam(model.parameters(), lr=1e-3)

# # mean-squared error loss
# criterion = nn.MSELoss()

# for epoch in range(epochs):
#     loss = 0
#     for batch_features, _ in train_loader:
#         # reshape mini-batch data to [N, 784] matrix
#         # load it to the active device
#         batch_features = batch_features.view(-1, 784).to(device)
        
#         # reset the gradients back to zero
#         # PyTorch accumulates gradients on subsequent backward passes
#         optimizer.zero_grad()
        
#         # compute reconstructions
#         outputs = model(batch_features)
        
#         # compute training reconstruction loss
#         train_loss = criterion(outputs, batch_features)
        
#         # compute accumulated gradients
#         train_loss.backward()
        
#         # perform parameter update based on current gradients
#         optimizer.step()
        
#         # add the mini-batch training loss to epoch loss
#         loss += train_loss.item()
    
#     # compute the epoch training loss
#     loss = loss / len(train_loader)
    
#     # display the epoch training loss
#     print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))