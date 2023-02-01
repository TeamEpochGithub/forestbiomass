import json

import numpy as np


def convert_values_to_json():
    # csv_path = r"test_corruptedness_values.csv"
    csv_path = r"test_intensity_values.csv"

    dicts = dict()
    with open(csv_path, "r") as f:
        data = f.readlines()
        for item in data:
            key = item[:8]
            print(key)
            item = item[11:-3]
            item_list = item.strip().split(",")
            dicts[key] = list(map(lambda x: float(x), item_list))

    print("normalizing...")
    b = json.dumps(normalize_dict(dicts))
    # f2 = open("test_corruptedness_values.json", "w")
    f2 = open("test_intensity_values.json", "w")
    f2.write(b)
    f2.close()


def normalize_dict(d):
    normalized_dict = {}

    means = []
    stds = []
    for key, values in d.items():
        for i in range(len(values)):
            column_mean = np.mean([d[k][i] if d[k][i] != -9999.0 else 0 for k in d.keys()])
            column_std = np.std([d[k][i] if d[k][i] != -9999.0 else 0 for k in d.keys()])
            means.append(column_mean)
            stds.append(column_std)
        break

    for key, values in d.items():
        print(key)
        normalized_values = []
        for i in range(len(values)):
            # column_mean = np.mean([d[k][i] if d[k][i] != -9999.0 else 0 for k in d.keys()])
            # column_std = np.std([d[k][i] if d[k][i] != -9999.0 else 0 for k in d.keys()])
            column_mean = means[i]
            column_std = stds[i]

            if column_std == 0:
                column_std = 1
            normalized_values.append((values[i] - column_mean) / column_std)
        normalized_dict[key] = normalized_values
    return normalized_dict


if __name__ == '__main__':
    convert_values_to_json()
