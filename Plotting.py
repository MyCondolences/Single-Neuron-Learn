import time
import math

import matplotlib.pyplot as plt
import pickle

# Load the relevant variables from the pickle file
with open('variables.pkl', 'rb') as f:
    data = pickle.load(f)

epochsarr = data['epochsarr']
min_epoch_errors_by_epoch = data['min_epoch_errors_by_epoch']
min_epoch_errors_by_epoch20 = data['min_epoch_errors_by_epoch20']
min_epoch_errors_by_epoch80 = data['min_epoch_errors_by_epoch80']
min_epoch_error_LRs_by_epoch = data['min_epoch_error_LRs_by_epoch']
weights = data['weights']
print_text = data['print_text']
len80 = data['len80']
len20 = data['len20']

for text in print_text:
    print(text)

print(weights)


def plot_err(epochs, errArr, minArr, text):
    min_error = min(errArr[epochs])
    min_index = errArr[epochs].index(min_error)
    minArr.append(min_error)

    plt.figure(figsize=(10, 6))
    plt.plot(min_epoch_error_LRs_by_epoch[epochs], errArr[epochs], marker='o')

    plt.plot(min_epoch_error_LRs_by_epoch[epochs][min_index], errArr[epochs][min_index], 'ro',
             label='Min Error')

    plt.annotate(f'Min LR={min_epoch_error_LRs_by_epoch[epochs][min_index]:.1e}',
                 (min_epoch_error_LRs_by_epoch[epochs][min_index], errArr[epochs][min_index]),
                 textcoords="offset points", xytext=(0, -15), ha='center', color='red')

    for i in range(len(errArr[epochs])):
        plt.annotate(f'LR={min_epoch_error_LRs_by_epoch[epochs][i]:.1e}',
                     (epochs, errArr[epochs][i]),
                     textcoords="offset points", xytext=(0, 10), ha='center')

    plt.xlabel('Learn Rate')
    plt.ylabel('Sum squared Error')
    plt.title(f'{text} vs Learning rate for {epochs} Epochs')
    plt.grid(True)
    plt.show()


def plot_avg(epochs, errArr, errArr2, minArr, text):

    plt.figure(figsize=(10, 6))
    plt.plot(min_epoch_error_LRs_by_epoch[epochs], [math.sqrt(x) / len20 for x in errArr[epochs]], marker='o', label='Error test')

    plt.plot(min_epoch_error_LRs_by_epoch[epochs],  [math.sqrt(x) / len80 for x in errArr2[epochs]], marker='x', label='Error '
                                                                                                            'train',
             color='orange')

    plt.xlabel('Learn Rate')
    plt.ylabel('Average Error')
    plt.title(f'{text} vs Learning rate for {epochs} Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()


minArr = []
for epochs in epochsarr:
    time.sleep(2)
    # tot error
    plot_err(epochs, min_epoch_errors_by_epoch, minArr, 'Total Error')
    time.sleep(2)
    # 20 error
    plot_err(epochs, min_epoch_errors_by_epoch20, [], 'Test Set Error')
    time.sleep(2)
    # 80 error
    plot_err(epochs, min_epoch_errors_by_epoch80, [], 'Training Set Error')
    time.sleep(2)

    plot_avg(epochs, min_epoch_errors_by_epoch20, min_epoch_errors_by_epoch80, [], 'Avg Set Error')

plt.figure(figsize=(10, 6))
plt.plot(epochsarr, minArr, marker='o')

for i, txt in enumerate(epochsarr):
    plt.annotate(f'{txt}', (epochsarr[i], minArr[i]), textcoords="offset points", xytext=(0, 10), ha='center')

plt.xlabel('Total Epochs')
plt.ylabel('Minimum Absolute Error in Epoch')
plt.title(f'Total Epochs vs. Minimum Absolute Error in Epoch for Epochs')
plt.grid(True)
plt.show()
