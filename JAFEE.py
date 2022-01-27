#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Subtract, Lambda
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
import keras.backend as K
from sklearn.model_selection import train_test_split
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import cv2
import itertools
import time


# In[2]:


labels = {
  "surprise": 0,
  "angry": 1,
  "happy": 2,
  "fear": 3,
  "disgust": 4,
  "sad": 5,
  "neutral": 6
}


# In[3]:


def abc(str1):
    for i in range (len(str1)-5,0,-1):
        try:
            z = int(str1[i])
        except:
            return str1[:i+1]


# In[17]:


def getImage(im,k):
    im = plt.imread('../Datasets/Facial-Expression-Recognization-using-JAFFE-master/jaffe/AllFiles/'+im.numpy()[k].decode('utf8'))
    im = im.astype('float32')
    im = im/255
    im=cv2.resize(im,(64,64))
    img = im.reshape((-1,64 * 64))
    return img


# In[5]:


data_path = '../Datasets/Facial-Expression-Recognization-using-JAFFE-master/id'
data_dir_list = os.listdir(data_path)

n_class = 7
img_data_list=[]
x = []
y = []
for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    combinations = list(itertools.combinations(range(len(img_list)), 2))
    for i in range (len(combinations)):
        imgName1 = img_list[combinations[i][0]]
        imgName2 = img_list[combinations[i][1]]
#         label1 = labels[abc(imgName1)]
#         label2 = labels[abc(imgName2)]
        label1 = tf.one_hot(labels[abc(imgName1)], n_class)
        label2 = tf.one_hot(labels[abc(imgName2)], n_class)
#         img1 = plt.imread(data_path + '/'+ dataset + '/'+ imgName1)
#         img2 = plt.imread(data_path + '/'+ dataset + '/'+ imgName2)
#         img1 = cv2.resize(img1,(128,128))
#         img2 = cv2.resize(img2,(128,128))
#         img1 = img1.astype('float32')
#         img1 = img1/255
#         img2 = img2.astype('float32')
#         img2 = img2/255
#         img1 = img1.reshape((-1, 128 * 128))
#         img2 = img2.reshape((-1, 128 * 128))
        x.append((imgName1,imgName2))
        y.append((label1,label2))
        



# img_data = img_data.reshape((-1, 128 * 128))


# In[6]:


len(x)


# In[7]:


#x = x[:-14]


# In[8]:


#y = y[:-14]


# In[9]:


# Building model
n_class = 7
input_dim = 4096
z_dim = 256



# Encoder
def make_encoder_model():
    inputs = tf.keras.Input(shape=(input_dim,), name='Original_input')
    x = tf.keras.layers.Dense(2048, activation='relu')(inputs)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    latent = tf.keras.layers.Dense(z_dim, activation='linear', name='Latent_variables')(x)
    observed = tf.keras.layers.Dense(n_class, activation='softmax', name='Observed_variables')(x)

    model = tf.keras.Model(inputs=inputs, outputs=[latent, observed], name='Encoder')
    return model


encoder = make_encoder_model()


# # In[ ]:


encoder.summary()


# In[10]:


# # Decoder
def make_decoder_model():
    inputted_latent = tf.keras.Input(shape=(z_dim,), name='Latent_variables')
    inputted_observed = tf.keras.Input(shape=(n_class,), name='Observed_variables')

    x = tf.keras.layers.concatenate([inputted_latent, inputted_observed], axis=-1)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(2048, activation='relu')(x)
    reconstruction = tf.keras.layers.Dense(input_dim, activation='linear', name='Reconstruction')(x)
    model = tf.keras.Model(inputs=[inputted_latent, inputted_observed], outputs=reconstruction, name='Decoder')
    return model


decoder = make_decoder_model()


# In[11]:


# Multipliers
alpha = 1000.0
beta = 100.0
gamma = 1000.0
zeta = 100

# Loss functions
# Reconstruction cost
mse_loss_fn = tf.keras.losses.MeanSquaredError()

# Supervised cost
cat_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)


# In[35]:


# Unsupervised cross-covariance cost
def xcov_loss_fn(latent, observed,batch_size):
    latent_centered = latent - tf.reduce_mean(latent, axis=0, keepdims=True)
    observed_centered = observed - tf.reduce_mean(observed, axis=0, keepdims=True)
    xcov_loss = 0.5 * tf.reduce_sum(
        tf.square(tf.matmul(latent_centered, observed_centered, transpose_a=True)))

    return xcov_loss


# In[12]:


optimizer = tf.keras.optimizers.Adam(lr=0.0001)


# In[14]:


batch_size = 8
n_epochs = 100

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=2)
train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.shuffle(buffer_size=1024)

#np.save('x_test.npy',x_test)
#np.save('y_test.npy',y_test)
# In[24]:


# Training step
# @tf.function
def train_on_batch(batch_x, batch_y):
    with tf.GradientTape() as tape:
        # Inference
        batch_x_1 = np.zeros((batch_size,64*64))
        batch_x_2 = np.zeros((batch_size,64*64))
        
        batch_y_1 = np.zeros((batch_size,7))
        batch_y_2 = np.zeros((batch_size,7))
        
        for i in range (batch_size):     
            batch_x_1[i] = getImage(batch_x[i],0)
            batch_x_2[i] = getImage(batch_x[i],1)

            batch_y_1[i] = batch_y[i].numpy()[0]
            batch_y_2[i] = batch_y[i].numpy()[1]
            
        batch_latent_1, batch_observed_1 = encoder(batch_x_1)
        batch_latent_2, batch_observed_2 = encoder(batch_x_2)
        batch_reconstruction_1 = decoder([batch_latent_1, batch_observed_1])
        batch_reconstruction_2 = decoder([batch_latent_2, batch_observed_2])
        
#         plt.figure()
#         print(batch_x_1)
#         print(batch_observed_1)
        # Loss functions
        recon_loss_1 = alpha * mse_loss_fn(batch_x_1, batch_reconstruction_1)
        recon_loss_2 = alpha * mse_loss_fn(batch_x_2, batch_reconstruction_2)
#         print(batch_reconstruction_1.shape)
#         plt.imshow(batch_latent_1)
#     plt.imshow(batch_latent_1)
        
        cat_loss_1 = beta * cat_loss_fn(batch_y_1, batch_observed_1)
        cat_loss_2 = beta * cat_loss_fn(batch_y_2, batch_observed_2)
        
        xcov_loss_1 = gamma * xcov_loss_fn(batch_latent_1, batch_observed_1, tf.cast(tf.shape(batch_x_1)[0], tf.float32))
        xcov_loss_2 = gamma * xcov_loss_fn(batch_latent_2, batch_observed_2, tf.cast(tf.shape(batch_x_2)[0], tf.float32))
    
        similarity_loss = zeta*mse_loss_fn(batch_latent_1,batch_latent_2)
        
        # Final loss function
        ae_loss = recon_loss_1 + recon_loss_2 + cat_loss_1 +cat_loss_2 + xcov_loss_1 + xcov_loss_2 + similarity_loss
               
    gradients = tape.gradient(ae_loss, encoder.trainable_variables + decoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables + decoder.trainable_variables))
    recon_loss = (recon_loss_1 + recon_loss_2)/2
    cat_loss = (cat_loss_1 + cat_loss_2)/2
    xcov_loss = (xcov_loss_1 + xcov_loss_2)/2
    return recon_loss, cat_loss, xcov_loss, similarity_loss


# In[25]:


for epoch in range(n_epochs):
    start = time.time()
    
    # Functions to calculate epoch's mean performance
    epoch_recon_loss_avg = tf.metrics.Mean()
    epoch_cat_loss_avg = tf.metrics.Mean()
    epoch_xcov_loss_avg = tf.metrics.Mean()
    epoch_sim_loss_avg = tf.metrics.Mean()

    for batch, (batch_x, batch_y) in enumerate(train_dataset):
        if(len(batch_x)<8):break
        recon_loss, cat_loss, xcov_loss, similarity_loss = train_on_batch(batch_x, batch_y)
        epoch_recon_loss_avg(recon_loss)
        epoch_cat_loss_avg(cat_loss)
        epoch_xcov_loss_avg(xcov_loss)
        epoch_sim_loss_avg(similarity_loss)
        

    epoch_time = time.time() - start
    print('{:3d}: {:.2f}s ETA: {:.2f}s  Reconstruction cost: {:.4f}  Supervised cost: {:.4f}  XCov cost: {:.4f}  Similarity cost: {:.4f}'
          .format(epoch + 1, epoch_time,
                  epoch_time * (n_epochs - epoch),
                  epoch_recon_loss_avg.result(),
                  epoch_cat_loss_avg.result(),
                epoch_xcov_loss_avg.result(),epoch_sim_loss_avg.result()))

    
encoder.save('Ronaks encoder.h5')
decoder.save('Ronaks decoder.h5')

