import csv


def load_data(fil):
    with open(fil, 'r') as file:
        reader = csv.reader(file)
        data = []
        next(reader)  # skips headings
        for row in reader:
            data.append([int(e) for e in row])
    return data


filename = 'SalData.csv'
data = load_data(filename)
evaldata = load_data('Evaluation.csv')


w = [1193.9377239174057, 299.5236463183904, -757.7700511441054, 331.3644041155769, 306.00036273508056,
     168.2301867942084, 212.34367222753707] # 50 000 epochs 1e-06
    #[1191.599185692986, 306.0175729297685, -769.8262691316755, 330.8762385423269, 286.1924404405571,
    # 171.73669847386768, 216.55635377077226]  # 500 epochs @5e-06


def output(x, weights):
    out = (weights[0] * x[0] + weights[1] * x[1] + weights[2] * x[2] +
           weights[3] * x[3] + weights[4] * x[4] + weights[5] * x[5] +
           weights[6] * x[6])

    return out



for j in range(len(evaldata)):
    result = output(evaldata[j], w)
    print(f'result: {result:.2f} from: {evaldata[j]}')
