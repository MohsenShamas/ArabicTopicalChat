import pandas
import csv
import re

with open("ArabicTopicalChat_clean.csv", "w", encoding="utf-8", newline='') as outfile:
    writer = csv.writer(outfile)
    ar_data = pandas.read_csv("translated_dataset/preprocessed.csv", encoding="utf-8")
    count=0
    for ar_line in ar_data.values:
        if not (re.search('[a-zA-Z]', ar_line[1]) or re.search('[a-zA-Z]', ar_line[0])):
            writer.writerow(ar_line)
            count+=1

