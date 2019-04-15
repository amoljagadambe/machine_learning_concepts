import cv2
import os

path = os.path.abspath(os.path.dirname(__name__))
filepath = path + '/files/Cristiano-Ronaldo.jpg'
img_read = cv2.imread(filepath)
resized_img = cv2.resize(img_read,(600,450))

cv2.imshow(filepath, resized_img)

cv2.waitKey(2000)

cv2.destroyAllWindows()