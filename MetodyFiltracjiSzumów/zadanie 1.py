import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.ndimage as snd
import numpy as np

img = mpimg.imread(r"C:\Inżynieria Biomedyczna inż\Semestr 7\Praca inżynierska\Obrazy OCT\CO_06\CO_06_SZIA_20230615_111254_Anterior_B-skan_Szeroki_L_14mm_12032x1_scan1.png")
filtr = np.array([[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]])

sobel = np.array([[1, 2, 1],
                  [0, 0, 0],
                  [-1, -2, -1]])

prewitt = np.array([[1, 1, 1],
                  [0, 0, 0],
                  [-1, -1, -1]])

obraz1 = snd.convolve(img[:,:,1], sobel)
obraz2 = snd.convolve(img[:,:,1], prewitt)
obraz3 = snd.gaussian_laplace(img, 0.5)
obraz4 = cv2.bilateralFilter(img,3,75,75)
obraz5 = snd.median_filter(img,3)
obraz6 = cv2.blur(img,(3,3))

con = snd.convolve(img[:,:,1],filtr)
con4 = snd.convolve(obraz4[:,:,1],filtr)
con5 = snd.convolve(obraz5[:,:,1],filtr)
con6 = snd.convolve(obraz6[:,:,1],filtr)

# cv2.imshow('obraz', obraz1)
# cv2.waitKey()
# cv2.imshow('obraz', obraz2)
# cv2.waitKey()
# cv2.imshow('obraz', obraz3)
# cv2.waitKey()
# cv2.imshow('obraz', obraz4)
# cv2.waitKey()
# cv2.imshow('obraz', obraz5)
# cv2.waitKey()
# cv2.imshow('obraz', obraz6)
# cv2.waitKey()
# cv2.imshow('obraz', con)
# cv2.waitKey()
# cv2.imshow('obraz', con4)
# cv2.waitKey()
# cv2.imshow('obraz', con5)
# cv2.waitKey()
# cv2.imshow('obraz', con6)
# cv2.waitKey()

print("obraz oryginalny - percentyl 99.9:")
print(np.percentile(con, 99.9))
print("obraz po filtracji bilateralnej - percentyl 99.9:")
print(np.percentile(con4, 99.9))
print("obraz po filtracji medianowej - percentyl 99.9:")
print(np.percentile(con5, 99.9))
print("obraz po filtracji uśredniającej - percentyl 99.9:")
print(np.percentile(con6, 99.9))
