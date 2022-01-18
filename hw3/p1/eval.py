'''
usage: python3 [pred.csv]
'''
import sys, csv
args = sys.argv

with open(args[1], mode='r') as pred:
    reader = csv.reader(pred)
    pred_dict = {rows[0]:rows[1] for rows in reader}

total_count = 0
correct_count = 0
for key, value in pred_dict.items():
    if value == 'label':
        continue
    if key[:key.find("_")] == value:
        correct_count += 1
    total_count += 1

accuracy = (correct_count / total_count) * 100
print('Accuracy: {}/{} ({}%)'.format(correct_count, total_count, accuracy))