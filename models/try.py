from PIL import Image
import numpy as np
import os
path = r"C:\Users\Team Epoch A\Documents\Epoch III\forestbiomass\data\imgs\train_agbm\0e174295_agbm.tif"
path1 = r"C:\Users\Team Epoch A\Documents\Epoch III\forestbiomass\data\imgs\test_agbm\a1f1bf0f_agbm.tif"
img = Image.open(path1)
img = np.array(img)
img = np.round(img)
non_zeros = np.count_nonzero(img)

ratio = (256*256-non_zeros)/(256*256)
print(ratio)
print(np.min(img))
print(np.max(img))
print(img)


data_path = r"C:\Users\Team Epoch A\Documents\Epoch III\forestbiomass\data\imgs\train_agbm"
ratios = []
# save_path = r"C:\Users\Team Epoch A\Documents\Epoch III\forestbiomass\data\imgs\test_agbm_new"
for file in os.listdir(data_path):
    img_path = os.path.join(data_path, file)
    img = Image.open(img_path)
    img = np.array(img)
    # img = np.round(img)
    non_zeros = np.count_nonzero(img)
    ratio = (256 * 256 - non_zeros) / (256 * 256)
    ratios.append(ratio)
ratios = np.array(ratios)
print(np.min(ratios))
print(np.max(ratios))
print(np.mean(ratios))
print(np.median(ratios))
#     imgmax = np.max(img)
#     img_new = np.clip(img, 0, imgmax)
#     im = Image.fromarray(img_new)
#     path_tmp = os.path.join(save_path, file)
#     im.save(path_tmp)

