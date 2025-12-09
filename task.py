import os
import cv2
import matplotlib.pyplot as plt


images = []

for archivo in os.listdir("images"):
        img = cv2.imread(os.path.join("images", archivo))
        images.append(img)

for im in images:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        plt.imshow(im)
        plt.show()

