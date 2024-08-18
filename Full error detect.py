import csv
import matplotlib.pyplot as plt
import random


def load_data(fil):
    with open(fil, 'r') as file:
        reader = csv.reader(file)
        data = []
        next(reader)  # skips headings
        for row in reader:
            data.append([int(e) for e in row])
    return data


def separate_columns(data):
    first_column = [row[0] for row in data]
    rest_columns = [row[1:] for row in data]
    return first_column, rest_columns


filename = 'SalData.csv'
data = load_data(filename)
evaldata = load_data('Evaluation.csv')

trainSal, traindata = separate_columns(data)

plt.plot(range(len(trainSal)), trainSal)
plt.grid(True)
plt.show()

ws = [[143.75385363351145, 297.86765962640527, -269.63520214520133, 339.52682562508426, 49.28071025360784,
       210.2388229456918, 51.73120113106255],  # 2 epoch @ 5e-07
      [16.89532430318019, 260.0076805872376, 4.967679330778441, 326.2163619883855, 12.84455437327249,
       221.30423788550016, 72.43206116094356],  # 2 epoch @ 5e-07
      [16.79732305829254, 261.79994430816055, 4.922276200066059, 327.05130019741404, 12.557616487660765,
       221.46962343331012, 71.5349031666169],  # 10 epoch 1e=07
      [16.76793966231305, 262.0700759910678, 4.957586712994048, 327.13575355521, 12.514341276132303,
       221.3929245759642, 71.44776970556303],  # 20 epoch @ 5e-08
      [1219.1805519748382, 297.71437096324354, -776.3491983333319, 345.15513506930165, 336.1164279274388,
       179.15134183239604, 217.98053356762273],  # 2000 epoch @5e-05
      [16.30438850059195, 247.51532446235748, 8.334786429368593, 326.8381227998703, 12.534378384639115,
       227.96370401344953, 76.40611629858108],  # 10 epochs @1e-07
      [1167.7512860804952, 280.45670495359246, -761.7619923299937, 332.66248923655587, 291.7776806340612,
       167.89163937134552, 193.53555817185736],  # 200 epochs @5e-05
      [1191.599185692986, 306.0175729297685, -769.8262691316755, 330.8762385423269, 286.1924404405571,
       171.73669847386768, 216.55635377077226],  # 500 epochs @5e-06
      [1188.9256928049654, 301.14338816786807, -775.5263307412458, 328.31947479874907, 317.739189609976,
       167.0489423120057, 214.85082815107486],  # 500 epochs @5e-06
      [1193.9409761866066, 301.73443922753285, -766.9241282936966, 327.99602485620244, 316.1834950569374,
       169.70270226475293, 212.35581156572417],  # 50 000 epochs @ 5e-08
      [285.1900858611275, 306.24535748931555, -433.2510446579596, 335.4383609496037, 75.56057324052912,
       213.13229768904029, 97.56389862753332]  # 1000 eopocks @1e-06 with cutoff
      ]

annotations = ['2 epoch @ 5e-07',
               '2 epoch @ 5e-07',
               '10 epoch @ 1e-07',
               '20 epoch @ 5e-08',
               '2000 epoch @ 5e-05',
               '10 epochs @ 1e-07',
               '200 epochs @ 5e-05',
               '500 epochs @ 5e-06',
               '500 epochs @ 5e-06',
               '50 000 epochs @ 5e-08',
               '1000 eopocks @1e-06 with cutoff 65000'
               ]


def output(x, weights):
    out = (weights[0] * x[0] + weights[1] * x[1] + weights[2] * x[2] +
           weights[3] * x[3] + weights[4] * x[4] + weights[5] * x[5] +
           weights[6] * x[6])

    return out


errort = []
for w in ws:
    error = 0
    predicted_results = []
    for j in range(len(traindata)):
        result = output(traindata[j], w)
        predicted_results.append(result)
        error += abs(trainSal[j] - result)
    errort.append(error)

plt.figure(figsize=(10, 6))
plt.plot(range(len(ws)), errort, 'o-', label='Total Error')

for i in range(len(annotations)):
    plt.annotate(annotations[i],
                 (i, errort[i]),
                 textcoords="offset points", xytext=(0, 10), ha='center')

plt.xlabel('Weights Index')
plt.ylabel('Total Error')
plt.title('Total Error Across Different Sets of Weights')
plt.legend()
plt.grid(True)
plt.show()

print(errort)
