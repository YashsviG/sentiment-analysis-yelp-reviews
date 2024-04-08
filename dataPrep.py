
import argparse
import json
import pandas as pd
from sklearn.model_selection import train_test_split

keys_to_remove = ["review_id", "business_id", "user_id", "date"]
outfile = "./cleaned_data.json"
# input_filename = "./test.json"


# Remove punctuations from text
def remove_punctuation(line: pd.DataFrame):
    punctuation = "\"#$%&'()*+,-./:;<=>@[\\]^_`{|}~.\n"
    replacement_mapping = {"&": "and", "/": " or ", "\\": "or", "<": "lt", ">": "gt"}
    translation_table = str.maketrans("", "", punctuation)

    for char, replacement in replacement_mapping.items():
        translation_table[ord(char)] = replacement

    line["text"] = line["text"].translate(translation_table)


def load_raw_json(input_filename):
    with open(input_filename, "r") as file, open(outfile, "a") as output_file:
        for line in file:
            data = json.loads(line)
            for key in keys_to_remove:
                data.pop(key)
            remove_punctuation(data)
            output_file.write(json.dumps(data, ensure_ascii=False) + "\n")
        output_file.close()
    file.close()


def split_data(output_file=outfile):
    print(f"Output file is {output_file}")
    df = pd.read_json(output_file, lines=True, chunksize=100)
    print(df)

    train_ratio = 0.8
    test_ratio = 0.10
    val_ratio = 0.10
    test_val_ratio = test_ratio / (test_ratio + val_ratio)

    for chunk in df:

        train_data, remaining_data = train_test_split(chunk, test_size=(1 - train_ratio))
        test_data, val_data = train_test_split(remaining_data, test_size=test_val_ratio)
        print("data has been split.")

        train_data.to_json("train_data.json", orient="records", lines=True, mode="a")
        test_data.to_json("test_data.json", orient="records", lines=True, mode="a")
        val_data.to_json("val_data.json", orient="records", lines=True, mode="a")


def main():
    parser = argparse.ArgumentParser(description="Data Prep Script")

    parser.add_argument("--input_raw_file", type=str, help="Raw json input filepath")
    parser.add_argument("--input_file", type=str, help="Process json input filepath")
    parser.add_argument("--split_data", action="store_true", help="Flag to split data")

    args = parser.parse_args()

    if args.split_data:
        if not args.input_file:
            print("No Input file provided to split, using default values")
            split_data()
        else:
            split_data(args.input_file)
    elif args.input_raw_file:
        load_raw_json(args.input_raw_file)
    else:
        print("No action specified. Use --split-data or provide --input_raw_file")


if __name__ == "__main__":
    main()
