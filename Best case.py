from main import load_data, output

filename = 'SalData.csv'
data = load_data(filename)
evaldata = load_data('Evaluation.csv')

w = [1193.9377239174057, 299.5236463183904, -757.7700511441054, 331.3644041155769, 306.00036273508056,
     168.2301867942084, 212.34367222753707]  # 50 000 epochs 1e-06
# [1191.599185692986, 306.0175729297685, -769.8262691316755, 330.8762385423269, 286.1924404405571,
# 171.73669847386768, 216.55635377077226]  # 500 epochs @5e-06

for j in range(len(evaldata)):
    result = output(evaldata[j], w)
    print(f'result: {result:.2f} from: {evaldata[j]}')
