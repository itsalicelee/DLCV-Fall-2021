import pandas as pd
import csv

cnt = 0
total = 0
with open('./src/output.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)

    for line in csv_reader:
        name, label = line[0], line[1]
        idx = name.find("_")
        total += 1
        if name[:idx] == label:
            cnt += 1
print(cnt)
print(total)
print("Acc", cnt/total)