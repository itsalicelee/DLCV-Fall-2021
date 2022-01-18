import sys, csv
import numpy as np
args = sys.argv

# read your prediction file
with open(args[1], mode='r') as pred:
    reader = csv.reader(pred)
    next(reader, None)  # skip the headers
    pred_dict = {int(rows[0]): np.array(rows[1:]).astype(int) for rows in reader}

# read ground truth data
with open(args[2], mode='r') as gt:
    reader = csv.reader(gt)
    next(reader, None)  # skip the headers
    gt_dict = {int(rows[0]): np.array(rows[1:]).astype(int) for rows in reader}

if len(pred_dict) != len(gt_dict):
    sys.exit("Test case length mismatch.")

episodic_acc = []
for key, value in pred_dict.items():
    if key not in gt_dict:
        sys.exit("Episodic id mismatch: \"{}\" does not exist in the provided ground truth file.".format(key))

    episodic_acc.append((gt_dict[key] == value).mean().item())

episodic_acc = np.array(episodic_acc)
mean = episodic_acc.mean()
std = episodic_acc.std()

print('Accuracy: {:.2f} +- {:.2f} %'.format(mean * 100, 1.96 * std / (600)**(1/2) * 100))
