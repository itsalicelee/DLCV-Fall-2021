import sys, csv
args = sys.argv

with open(args[1], mode='r') as pred:
    reader = csv.reader(pred)
    pred_dict = {rows[0]:rows[1] for rows in reader}

with open(args[2], mode='r') as gt:
    reader = csv.reader(gt)
    gt_dict = {rows[0]:rows[1] for rows in reader}

total_count = 0
correct_count = 0
for key, value in pred_dict.items():
    if key not in gt_dict:
        sys.exit("Item mismatch: \"{}\" does not exist in the provided ground truth file.".format(key))
    if value == 'label':
        continue
    if gt_dict[key] == value:
        correct_count += 1
    total_count += 1

accuracy = (correct_count / total_count) * 100
print('Accuracy: {}/{} ({}%)'.format(correct_count, total_count, accuracy))