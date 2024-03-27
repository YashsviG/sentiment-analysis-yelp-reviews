import json
import pandas as pd
from sklearn.model_selection import train_test_split

keys_to_remove = ["review_id", "business_id", "user_id", "date"]
outfile = './data.json'

with open("./test.json", "r") as file, open(outfile, "a") as output_file:
    for line in file:
        data = json.loads(line)
        for key in keys_to_remove:
            data.pop(key)
        output_file.write(json.dumps(data, ensure_ascii=False)+'\n')
    output_file.close()

file.close()

df = pd.read_json(outfile, lines=True)
train_ratio = 0.8
test_ratio = 0.10
val_ratio = 0.10
test_val_ratio = test_ratio / (test_ratio + val_ratio)

train_data, remaining_data = train_test_split(df, test_size=(1-train_ratio))
test_data, val_data = train_test_split(remaining_data, test_size=test_val_ratio)

train_data.to_json("train_data.json", orient="records", lines=True)
test_data.to_json("test_data.json", orient="records", lines=True)
val_data.to_json("val_data.json", orient="records", lines=True)