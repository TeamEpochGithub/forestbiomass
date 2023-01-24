import os

list = os.listdir("imgs/train_features")

patches = []
for i in list:
    patches.append(i[:-10])

print(set(patches))