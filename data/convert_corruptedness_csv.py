import json

csv_path = r"test_corruptedness_values.csv"

dicts = dict()
with open(csv_path, "r") as f:
    data = f.readlines()
    print(data[0])
    for item in data:
        key = item[:8]
        item = item[11:-3]
        item_list = item.strip().split(",")
        dicts[key] = list(map(lambda x:float(x), item_list))

b = json.dumps(dicts)
f2 = open("test_corruptedness_values.json", "w")
f2.write(b)
f2.close()

