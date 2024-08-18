import csv
import pickle

import matplotlib.pyplot as plt
import random

print_text = []


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


def remove_outliners(data, cutoff=float('inf')):
    newData = []
    for row in data:
        if row[0] < cutoff:
            newData.append(row)

    return newData


filename = 'SalData.csv'
raw_data = load_data(filename)
data = remove_outliners(raw_data)
data_80, data_20 = split_data(data)
len80 = len(data_80)
len20 = len(data_20)

train80Sal, train80data = separate_columns(data_80)
train20Sal, train20data = separate_columns(data_20)

# Salary,Educa,NrSuperv,NrPosition,Resp%,NumChil,Age,YearsExp


Learn = [0.0001, 0.00005, 0.00001, 0.000005, 0.000001, 0.0000005, 0.0000001, 0.00000005]


# weights = [0, 0 ,0 ,0 ,0, 0, 0]


def output(x):
    out = (weights[0] * x[0] + weights[1] * x[1] + weights[2] * x[2] +
           weights[3] * x[3] + weights[4] * x[4] + weights[5] * x[5] +
           weights[6] * x[6])

    return out


def update_weights(inputs, error, LR):
    global weights
    for i in range(len(weights)):
        weights[i] += LR * error * inputs[i]  # Update weights


epochsarr = [1, 2, 5, 10, 20, 50, 100, 200, 500, 750, 1000, 50000]
min_epoch_errors_by_epoch20 = {epoch: [] for epoch in epochsarr}
min_epoch_errors_by_epoch80 = {epoch: [] for epoch in epochsarr}

min_epoch_errors_by_epoch = {epoch: [] for epoch in epochsarr}
min_epoch_error_LRs_by_epoch = {epoch: [] for epoch in epochsarr}
epoch_error = 0
for LR in Learn:
    for epochs in epochsarr:
        print(f"Learning rate: {LR}  Epochs : {epochs}")
        print_text.append(f"Learning rate: {LR}  Epochs : {epochs}")

        weights = [0.5 for _ in range(7)]
        min_error = float('inf')
        for epoch in range(epochs):
            epoch_error = 0
            error = 0
            for i in range(len(train80data)):
                result = output(train80data[i])
                error = train80Sal[i] - result
                epoch_error += error
                update_weights(train80data[i], error, LR)

        # min_epoch_errors_by_epoch[epochs].append(epoch_error)
        # min_epoch_error_LRs_by_epoch[epochs].append(LR)

        errort = 0
        for i in range(len(train20data)):
            result = output(train20data[i])
            errort += abs(train20Sal[i] - result)

        min_epoch_errors_by_epoch20[epochs].append(errort ** 2)

        errort2 = 0
        for i in range(len(train80data)):
            result = output(train80data[i])
            errort2 += abs(train80Sal[i] - result)

        min_epoch_errors_by_epoch80[epochs].append(errort2 ** 2)

        min_epoch_errors_by_epoch[epochs].append(errort + errort2)
        min_epoch_error_LRs_by_epoch[epochs].append(LR)

        print(f"Weights : {weights}")
        print_text.append(f"Weights : {weights}")

# save data
with open('variables.pkl', 'wb') as f:
    pickle.dump({
        'epochsarr': epochsarr,
        'min_epoch_errors_by_epoch': min_epoch_errors_by_epoch,
        'min_epoch_errors_by_epoch20': min_epoch_errors_by_epoch20,
        'min_epoch_errors_by_epoch80': min_epoch_errors_by_epoch80,
        'min_epoch_error_LRs_by_epoch': min_epoch_error_LRs_by_epoch,
        'weights': weights,
        'print_text': print_text,
        'len80': len80,
        'len20': len20
    }, f)

print('training complete')
