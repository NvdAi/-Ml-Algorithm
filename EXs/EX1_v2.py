import cv2
import matplotlib.pyplot as plt
import numpy as np

def Histogram(img, nh):
    row, col, depth = img.shape

    hist_table = np.zeros(nh, np.int32)
    bins = np.arange(0,256,255/nh)
    for r in range(row):
        for c in range(col):            

            mean_of_pixels = sum(img[r][c])/depth
            for i in range(len(hist_table)):
                start = bins[i]
                stop = bins[1+i]
                    
                T1 = start <= mean_of_pixels
                T2 = mean_of_pixels < stop

                if stop == bins[-1]:
                    print(stop)
                    T2 = mean_of_pixels <= stop

                if T1 and T2 :
                    hist_table[i]+=1
                    break
        
    print(hist_table)
    return hist_table


img_path = "../A1.jpg"
NH = 255

img = cv2.imread(img_path)
vals = img.mean(axis=2).flatten()
b, bins, patches = plt.hist(vals, NH)
print(b)
print("==========================================================")
# My histogram function
hist_table = Histogram(img, NH)
x_axes = np.arange(0, 255, 255/NH)
plt.plot(x_axes, hist_table)
plt.show()