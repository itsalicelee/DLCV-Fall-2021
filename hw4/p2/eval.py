import sys, csv
import numpy as np
import pandas as pd
import re
args = sys.argv

# read your prediction file
df = pd.read_csv(args[1])
filenames = df["filename"].to_list()
labels = df["label"].to_list()

cnt = 0
for i in range(len(filenames)):
    filename, label = filenames[i], labels[i]
    d  = re.search(r"\d", filename)
    pred = filename[:d.start()]
    if pred == label:
        cnt += 1

print("Accuracy:{:.4f} ({}/{})".format(cnt/len(df)*100, cnt, len(df)))


    