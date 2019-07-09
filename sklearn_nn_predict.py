
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
get_ipython().run_line_magic('pylab', 'inline')
from skimage.measure import compare_ssim
from skimage.measure import structural_similarity as ssim
from sklearn.neural_network import MLPClassifier

#import imagehash
from PIL import Image
import pickle


# In[3]:


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


# In[10]:


filename = 'trained_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

test_img1 = "data/test/066/08_066.png"
#test_img2 = "data/test/066/03_066.png"
test_img2 = "data/test/066_forg/03_0101066.PNG"


test_img1 = cv2.imread(test_img1)
test_img1 = cv2.resize(test_img1,(28,28))
test_img1 = cv2.cvtColor(test_img1,cv2.COLOR_BGR2GRAY)

test_img2 = cv2.imread(test_img2)
test_img2 = cv2.resize(test_img2,(28,28))
test_img2 = cv2.cvtColor(test_img2,cv2.COLOR_BGR2GRAY)

(score, diff) = compare_ssim(test_img1, test_img2, full=True)
rmse = mse(test_img1 , test_img2)

im1 = test_img1.flatten().tolist()
im2 = test_img2.flatten().tolist()


test_x = []
test_x.append([score]+[rmse]+im1+im2)
predicted_res = loaded_model.predict(test_x)
print(predicted_res)
