import cv2
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt

input_file = '../../data/cropped/IMG_5158_w100_ds2_i0_x887_y407_.JPG'

original = cv2.imread(input_file).astype(np.uint8)
img=original.copy()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#cv2.imshow('image',img)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.imshow(img)
u = img.mean()
ax2.imshow(img-u)

eq_hist = exposure.equalize_adapthist(img)

ax3.imshow(eq_hist)

plt.show()