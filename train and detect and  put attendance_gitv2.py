#!/usr/bin/env python
# coding: utf-8

# In[2]:

#################   importing library
import cv2
import os
import keras
import numpy as np


##################  training model


# fetching the image from the location
# In this case the location is  'C:\Users\SHREYAS\Desktop\NewData\face111_v1'


newlocation = r'C:\Users\SHREYAS\Desktop\NewData\face111_v1'


# In this all the images are numbered from '0' to '17' with respect to the person mentioned :
# dummynames is used to store the no from '0' to '17':

dummynames = []

for i in os.listdir(newlocation):
    if(i.split('_')[0]) == 'person1' :
        dummynames.append(1)
    
    elif(i.split('_')[0]) == 'person2' :
        dummynames.append(2)

    elif(i.split('_')[0]) == 'person3' :
        dummynames.append(3)
    
    elif(i.split('_')[0]) == 'person4' :
        dummynames.append(4)
    
    elif(i.split('_')[0]) == 'person5' :
        dummynames.append(5)
            
    elif(i.split('_')[0]) == 'person6' :
        dummynames.append(6)

    elif(i.split('_')[0]) == 'person7' :
        dummynames.append(7)
    
    elif(i.split('_')[0]) == 'person8' :
        dummynames.append(8)
    
    elif(i.split('_')[0]) == 'person9' :
        dummynames.append(9)
        
    elif(i.split('_')[0]) == 'person10' :
        dummynames.append(10)

    elif(i.split('_')[0]) == 'person11' :
        dummynames.append(11)
    
    elif(i.split('_')[0]) == 'person12' :
        dummynames.append(12)
    
    elif(i.split('_')[0]) == 'person13' :
        dummynames.append(13)
        
    elif(i.split('_')[0]) == 'person14' :
        dummynames.append(14)

    elif(i.split('_')[0]) == 'person15' :
        dummynames.append(15)
    
    elif(i.split('_')[0]) == 'person16' :
        dummynames.append(16)
    
    elif(i.split('_')[0]) == 'person17' :
        dummynames.append(17)


# Extraction of features :

import cv2
feature = []
for i in os.listdir(newlocation):
    f =  cv2.imread(os.path.join(newlocation,i))
    f1 = cv2.resize(f,(100,100))
    feature.append(f1)


# The features are reduced to values from '0' to '1':


x = np.array(feature)/255
y = np.array(dummynames)


# In[10]:


x[11].shape


# In[11]:
# plotting the image of x[11]

import matplotlib.pyplot as plt
plt.imshow(x[11])
plt.show()


# In[12]:


import tensorflow as tf


# In[13]:

####################   adding cnn    ###################

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation = 'relu'))
#model.add(tf.keras.layers.Dense(80,activation = 'relu'))
model.add(tf.keras.layers.Dense(48,activation = 'relu'))
#model.add(tf.keras.layers.Dense(32,activation = 'relu'))
model.add(tf.keras.layers.Dense(18,activation = 'softmax'))


# In[14]:


model.compile(loss = 'sparse_categorical_crossentropy',optimizer = 'sgd',metrics=['accuracy'])


# Splitting the data in training and testing set :


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.10)


# In[195]:

##########   training the model 

model.fit(xtrain,ytrain,epochs = 30)

##########   model trained 


# addhere is used for attendance purpose
addhere = np.argmax(model.predict(xtest[2].reshape(1,100,100,3)))
addhere




# calculating the effeciency ------------------
pred = model.predict(xtest)
p = []
for i in pred:
    p.append(np.argmax(i))



######################   percentage 
(p == ytest).sum() / len(xtest)



######################## asking for image .  to verify and put attendance


# In[ ]:


detectlocation = r'C:\Users\SHREYAS\Desktop\New folder\1stnormal.jpg'



    detectf =  cv2.imread(os.path.join(detectlocation))
    detectf1 = cv2.resize(detectf,(100,100))
    detectfeature = detectf1
    detectfeature


detectx = detectfeature/255

detectx

addhere = np.argmax(model.predict(detectx.reshape(1,100,100,3)))
addhere



#################### store or put attendance in excel file in csv format



import pandas as pd
attendance = pd.read_csv(r'C:\Users\SHREYAS\Desktop\report_attendance.csv')

attendance.head()


# to add a count to addhere. addhere contains number to which that attendance belongs
attendance.loc[attendance['PersonNO'] == addhere,['count']] += 1


attendance.head()

attendance.to_csv(r'C:\Users\SHREYAS\Desktop\report_attendance.csv')

