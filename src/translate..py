import pandas
import csv
from tqdm import tqdm
from googletrans import Translator

translator = Translator()

with open("translated_dataset/translated.csv", "w", encoding="utf-8", newline='') as translation_file:
    csv_writer = csv.writer(translation_file)
    data = pandas.read_csv("dataset/topical_chat.csv")
    header = data.columns.tolist()
    rows = data.values
    # count = 0
    # csv_writer.writerow(header)
    for row in tqdm(rows):
        if row[1] == " ": 
            csv_writer.writerow([row[0], " ", row[2]])
            continue
        
        while True:
            try:
                row[1]  = translator.translate(row[1], dest='ar').text
                break
            except Exception as e:
                print("Error occured:",e.__traceback__)
        csv_writer.writerow(row)
        # count+=1
        # if count == 500: break
        
        # print(row)
        





