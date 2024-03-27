import json
import io

keys_to_remove = ["review_id", "business_id", "user_id", "date"]
outfile = './data.json'

with open("./test.json", "r") as file, open(outfile, "a") as output_file:
    for line in file:
        data = json.loads(line)
        for key in keys_to_remove:
            data.pop(key)
        output_file.write(json.dumps(data)+'\n')
    output_file.close()
    
