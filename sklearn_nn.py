
# coding: utf-8

# In[11]:


import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd

from skimage.measure import compare_ssim
#from skimage.measure import structural_similarity as ssim
from sklearn.neural_network import MLPClassifier

import imagehash
from PIL import Image
import pickle


# In[12]:


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


# In[3]:


data = pd.read_csv("data/train_data.csv")

data.insert(2, 'MSE', '0')
data.insert(3, 'SSIM', '0')

data.insert(4, 'org_image', '')
data.insert(5, 'cmp_image', '')

data.insert(6, 'org_image_flat', '')
data.insert(7, 'cmp_image_flat', '')

data = data.iloc[np.random.permutation(len(data))]

#data = data[:100]
X = data.iloc[:,:-1].values
T = data.iloc[:,-1].values


X[:,0] = 'data/train/'+X[:,0]
X[:,1] = 'data/train/'+X[:,1]

#original images
for i in range(X.shape[0]):
    img = cv2.imread(X[i,0])
    #print(img)
    img = cv2.resize(img,(28,28))
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #gray_img = cv2.resize(gray_img,(28,28))
    X[i,4] = gray_img

#comparison images
for i in range(X.shape[0]):
    img = cv2.imread(X[i,1])
    #print(img)
    img = cv2.resize(img,(28,28))
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #gray_img = cv2.resize(gray_img,(128,128))
    X[i,5] = gray_img

for i in range(X.shape[0]):
    #SSIM values
    (score, diff) = compare_ssim(X[i,4], X[i,5], full=True)
    X[i,2] = score

    #MSE values
    X[i,3] = mse(X[i,4], X[i,5])

    X[i,6] = X[i,4].flatten().tolist()
    X[i,7] = X[i,5].flatten().tolist()

    #X[i,1] = X[i,1].flatten()
#print(X)


# In[13]:


x_input = []

for i in range(X.shape[0]):
    append_lst = []
    #append_lst.append()
    #append_lst.append()
    append_lst = [X[i,2]] + [X[i,3]] + X[i,6] + X[i,7]
    x_input.append(append_lst)

#print(x_input)


# In[14]:


y = T.tolist()


# In[15]:


clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(x_input, y)

filename = 'trained_model.sav'
pickle.dump(clf, open(filename, 'wb'))


# In[22]:


test_img1 = "data/train/068/09_068.png"
#test_img2 = "data/train/068/09_068.png"
test_img2 = "data/train/068/02_068.png"

'''test_img1 = "data/test/066/08_066.png"
test_img2 = "data/test/066_forg/04_0211066.PNG"'''


test_img1 = cv2.imread(test_img1)
test_img1 = cv2.resize(test_img1,(28,28))
test_img1 = cv2.cvtColor(test_img1,cv2.COLOR_BGR2GRAY)

test_img2 = cv2.imread(test_img2)
test_img2 = cv2.resize(test_img2,(28,28))
test_img2 = cv2.cvtColor(test_img2,cv2.COLOR_BGR2GRAY)

(score, diff) = compare_ssim(test_img1, test_img2, full=True)
print(score)
rmse = mse(test_img1 , test_img2)
print(rmse)
im1 = test_img1.flatten().tolist()
im2 = test_img2.flatten().tolist()


test_x = []
test_x.append([score]+[rmse]+im1+im2)
clf.predict(test_x)


# In[23]:


loaded_model = pickle.load(open(filename, 'rb'))

test_img1 = "data/test/066/08_066.png"
test_img2 = "data/test/066_forg/04_0211066.PNG"


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
loaded_model.predict(test_x)
