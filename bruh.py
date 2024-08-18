import csv
import random
import numpy as np


def load_data(fil):
    with open(fil, 'r') as file:
        reader = csv.reader(file)
        data = []
        next(reader)  # skips headings
        for row in reader:
            data.append([int(e) for e in row])
    return data

def split_data(data, split_ratio=0.8):
    random.shuffle(data)
    split_index = int(len(data) * split_ratio)
    return data[:split_index], data[split_index:]

def separate_columns(data):
    first_column = [row[0] for row in data]
    rest_columns = [row[1:] for row in data]
    return first_column, rest_columns

def normalize_data(data):
    """Normalize the features to the range [0, 1]."""
    data = np.array(data)
    min_values = data.min(axis=0)
    max_values = data.max(axis=0)
    return (data - min_values) / (max_values - min_values)

filename = 'SalData.csv'
data = load_data(filename)
data_80, data_20 = split_data(data)

train80Sal, train80data = separate_columns(data_80)
train20Sal, train20data = separate_columns(data_20)

# Normalize the training data
train80data = normalize_data(train80data)

print(f"80% Data: {len(train80data)} records")
print(f"20% Data: {len(train20data)} records")

def clip_value(value, min_value=-1e10, max_value=1e10):
    return max(min(value, max_value), min_value)

def loss_function(y_i, y_ih):
    # Clip the values to prevent overflow
    y_i_clipped = clip_value(y_i)
    y_ih_clipped = clip_value(y_ih)

    # Print values for debugging

    return (y_i_clipped - y_ih_clipped) ** 2

LR = 0.01
bias = 1
weights = [random.random() for _ in range(7)]
print(f"Initial weights: {weights}")



def update_weights(inputs, error):
    global weights, bias
    bias += LR * error  # Update bias based on the error
    for i in range(len(weights)):
        weights[i] += LR * error * inputs[i]  # Update weights

epochs = 1000
for epoch in range(epochs):
    for i in range(len(train80data)):
        result = output(train80data[i])
        error = train80Sal[i] - result  # Calculate the prediction error
        update_weights(train80data[i], error)  # Use the raw error for updating weights

# Normalize the test data
train20data = normalize_data(train20data)

for i in range(len(train20data)):
    result = output(train20data[i])
    error = loss_function(train20Sal[i], result)
    print(f"Output: {result}, Target: {train20Sal[i]}")

print(f"Final weights: {weights}")
print(f"Final bias: {bias}")
