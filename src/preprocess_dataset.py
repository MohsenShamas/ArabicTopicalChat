import csv
import pandas
from tqdm import tqdm


PATH = "translated_dataset/translated.csv"
OUT_PATH = "translated_dataset/preprocessed.csv"

# with open(PATH, "r", encoding="utf-8") as dataFile:
#     lines = dataFile.read().split("\n")
#     print(lines[0])

df = pandas.read_csv(PATH)
with open(OUT_PATH, "w", encoding="utf-8", newline='') as outputFile:
    writer = csv.writer(outputFile)
    writer.writerow(["context", "response", "sentiment"])
    input_rows = df.values
    conversation_id = 1
    
    for ind in tqdm(range(len(input_rows)-1)):
        
        this_row = input_rows[ind]
        next_row = input_rows[ind + 1]
        if next_row[0] == conversation_id:
            output = [this_row[1], next_row[1], next_row[2]]
            writer.writerow(output)
            
        else:
            conversation_id+=1
    
