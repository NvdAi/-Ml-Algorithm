import cv2
import matplotlib.pyplot as plt
import numpy as np

def Histogram(img_pat, nh):
    img = cv2.imread(img_path)
    hist_table = np.zeros(256, np.int32)

    row, col, depth = img.shape
    for r in range(row):
        for c in range(col):
            mean_of_pixels = int(sum(img[r][c])/depth)
            hist_table[mean_of_pixels]+=1
    print(hist_table)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    hist_table_final = np.zeros(nh, np.int32)
    width = 255/nh
    # print(width)
    bins = np.arange(0,256,width)
    print(bins)
    for i in range(nh):
        start = int(bins[i])
        stop = int(bins[i+1])
        # print(start,stop)
        hist_table_final[i] = np.sum(hist_table[start:stop])
        if stop == int(bins[-1]):
            break

    hist_table_final[-1] = hist_table[-1]+hist_table_final[-1]
    print(hist_table_final)
    return hist_table


img_path = "../A1.jpg"
NH = 255

im = cv2.imread(img_path)

vals = im.mean(axis=2).flatten()
b, bins, patches = plt.hist(vals,NH)
print("=====================================================")
print(b)
print(bins)
print("=============================================")
# My histogram function
hist_table = Histogram(img_path,NH)
step = 255/NH
x_axes = np.arange(0,256,step)
plt.plot(x_axes,hist_table)
plt.xticks(np.arange(0,255,255/NH))
plt.show()

