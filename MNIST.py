#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[2]:


import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys



# In[3]:


# Set random seed
tf.random.set_seed(1)


# In[4]:


print("Loading data...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Flatten the dataset
x_train = x_train.reshape((-1, 28 * 28))
x_test = x_test.reshape((-1, 28 * 28))

batch_size = 256

# Create the training database iterator
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024)
train_dataset = train_dataset.batch(batch_size)


# In[5]:


# Building model
n_class = 10
input_dim = 784
z_dim = 2


# Encoder
def make_encoder_model():
    inputs = tf.keras.Input(shape=(input_dim,), name='Original_input')
    x = tf.keras.layers.Dense(500, activation='relu')(inputs)
    x = tf.keras.layers.Dense(500, activation='relu')(x)
    latent = tf.keras.layers.Dense(z_dim, activation='linear', name='Latent_variables')(x)
    observed = tf.keras.layers.Dense(n_class, activation='softmax', name='Observed_variables')(x)

    model = tf.keras.Model(inputs=inputs, outputs=[latent, observed], name='Encoder')
    return model


encoder = make_encoder_model()


# In[ ]:


encoder.summary()
# In[6]:


# Decoder
def make_decoder_model():
    inputted_latent = tf.keras.Input(shape=(z_dim,), name='Latent_variables')
    inputted_observed = tf.keras.Input(shape=(n_class,), name='Observed_variables')

    x = tf.keras.layers.concatenate([inputted_latent, inputted_observed], axis=-1)
    x = tf.keras.layers.Dense(500, activation='relu')(x)
    x = tf.keras.layers.Dense(500, activation='relu')(x)
    reconstruction = tf.keras.layers.Dense(input_dim, activation='linear', name='Reconstruction')(x)
    model = tf.keras.Model(inputs=[inputted_latent, inputted_observed], outputs=reconstruction, name='Decoder')
    return model


decoder = make_decoder_model()


# In[ ]:


decoder.summary()


# In[7]:


# Multipliers
alpha = 1.0
beta = 10.0
gamma = 10.0

# Loss functions
# Reconstruction cost
mse_loss_fn = tf.keras.losses.MeanSquaredError()

# Supervised cost
cat_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)


# In[8]:


# Unsupervised cross-covariance cost
def xcov_loss_fn(latent, observed, batch_size):
    latent_centered = latent - tf.reduce_mean(latent, axis=0, keepdims=True)
    observed_centered = observed - tf.reduce_mean(observed, axis=0, keepdims=True)
    xcov_loss = 0.5 * tf.reduce_sum(
        tf.square(tf.matmul(latent_centered, observed_centered, transpose_a=True) / batch_size))

    return xcov_loss


# In[9]:


optimizer = tf.keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)


# In[10]:


# Training step
@tf.function
def train_on_batch(batch_x, batch_y):
    with tf.GradientTape() as tape:
        # Inference
        batch_latent, batch_observed = encoder(batch_x)
        batch_reconstruction = decoder([batch_latent, batch_observed])

        # Loss functions
        recon_loss = alpha * mse_loss_fn(batch_x, batch_reconstruction)
        cat_loss = beta * cat_loss_fn(tf.one_hot(batch_y, n_class), batch_observed)
        xcov_loss = gamma * xcov_loss_fn(batch_latent, batch_observed, tf.cast(tf.shape(batch_x)[0], tf.float32))

        # Final loss function
        ae_loss = recon_loss + cat_loss + xcov_loss

    gradients = tape.gradient(ae_loss, encoder.trainable_variables + decoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables + decoder.trainable_variables))
    return recon_loss, cat_loss, xcov_loss


# In[11]:


n_epochs = 100
for epoch in range(n_epochs):
    start = time.time()

    # Functions to calculate epoch's mean performance
    epoch_recon_loss_avg = tf.metrics.Mean()
    epoch_cat_loss_avg = tf.metrics.Mean()
    epoch_xcov_loss_avg = tf.metrics.Mean()

    for batch, (batch_x, batch_y) in enumerate(train_dataset):
        recon_loss, cat_loss, xcov_loss = train_on_batch(batch_x, batch_y)

        epoch_recon_loss_avg(recon_loss)
        epoch_cat_loss_avg(cat_loss)
        epoch_xcov_loss_avg(xcov_loss)

    epoch_time = time.time() - start
    print('{:3d}: {:.2f}s ETA: {:.2f}s  Reconstruction cost: {:.4f}  Supervised cost: {:.4f}  XCov cost: {:.4f}'
          .format(epoch + 1, epoch_time,
                  epoch_time * (n_epochs - epoch),
                  epoch_recon_loss_avg.result(),
                  epoch_cat_loss_avg.result(),
                  epoch_xcov_loss_avg.result()))


# In[12]:


acc_fn = tf.keras.metrics.CategoricalAccuracy()

z_test, observed_test = encoder(x_test)
acc_test = acc_fn(tf.one_hot(y_test, n_class), observed_test)

print('Test set accuracy: {:.2f}'.format(100 * acc_test))


# Visualizating latent space
plt.figure()
plt.xlabel('z1')
plt.ylabel('z2')
plt.scatter(z_test.numpy()[:, 0], z_test.numpy()[:, 1], alpha=0.1)
plt.show()


# Sampling latent space

# Figure 3b and c
ys = np.repeat(np.arange(10), 9).astype('int32')
zs = np.tile(np.linspace(-0.5, 0.5, 9), 10).astype('float32')
z1s = np.vstack([zs, np.zeros_like(zs)]).T
z2s = np.vstack([np.zeros_like(zs), zs]).T

reconstructions_z1 = decoder([z1s, tf.one_hot(ys, n_class)]).numpy()
reconstructions_z2 = decoder([z2s, tf.one_hot(ys, n_class)]).numpy()

im1 = reconstructions_z1.reshape(10, 9, 28, 28).transpose(1, 2, 0, 3).reshape(9 * 28, 10 * 28)
plt.imshow(im1, cmap=plt.cm.gray)
plt.xlabel('Identity info')
plt.ylabel('Varying latent variable, z1')
plt.show()

im2 = reconstructions_z2.reshape(10, 9, 28, 28).transpose(1, 2, 0, 3).reshape(9 * 28, 10 * 28)
plt.imshow(im2, cmap=plt.cm.gray)
plt.xlabel('Identity info')
plt.ylabel('Varying latent variable, z2')
plt.show()
