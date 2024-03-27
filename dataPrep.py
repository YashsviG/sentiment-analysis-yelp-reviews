import json
import io

with open("./yelp_academic_dataset_review.json", "r") as file:
    data = json.load(file)
    keys_to_remove = ["review_id", "business_id", "user_id", "date"]
    for key in keys_to_remove:
        data.pop(key)

    outfile = io.open('data.json','a')

    json.dump(data, fp= outfile, indent=4)
