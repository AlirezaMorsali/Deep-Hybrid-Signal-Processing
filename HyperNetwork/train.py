import os
import numpy as np
import tensorflow as tf

from utils import *
from models import *






## Generate Data
samples = generate_sample()

random_channel = []
for counter in range(10): # OOM
    random_channel.append(mu_mm_ch_train(128, 8, NN=2000))
random_channel = np.concatenate(random_channel, axis=0)



n_train_channel = int(0.9 * len(random_channel))
train_channels = random_channel[:n_train_channel]
test_channels = random_channel[n_train_channel:]



## Create Network
base_model = base_network()
hyper_model = hyper_network()




## Train Process
optimizer = tf.keras.optimizers.Adam(1e-4)
loss_fn = tf.keras.losses.MeanSquaredError()

kernel1 = base_model.layers[-5].kernel
kernel2 = base_model.layers[-4].kernel


Epochs = 1
batch_size = 4 ** 8
loss_tracker = tf.keras.metrics.Mean()
Total_loss_tracker = tf.keras.metrics.Mean()


for epoch in range(Epochs):

    X = np.concatenate([np.real(samples), np.imag(samples)], axis=1)

    flag = 0
    pbar = tqdm(train_channels)
    for train_channel in pbar:

        train_channel = SVD(train_channel)

        real_y = generate_real_target(samples, train_channel)
        Y = np.concatenate([np.real(real_y), np.imag(real_y)], axis=1)
        Channel = np.stack([np.real(train_channel), np.imag(train_channel)], axis=2)[None, ...]

        for counter in range(int(4 ** 8 / batch_size)):
            loss_tracker.reset_states()
            x_batch, y_batch =  X[counter * batch_size:(counter + 1) * batch_size], Y[counter * batch_size:(counter + 1) * batch_size]
            with tf.GradientTape() as tape:
                generated_parameters = hyper_model(Channel)
                parameterize_model(base_model, generated_parameters)

                preds = base_model(x_batch)
                loss = loss_fn(y_batch, preds)
                loss_tracker.update_state(loss)
            grads = tape.gradient(loss, hyper_model.trainable_weights + base_model.trainable_weights[:-2])
            optimizer.apply_gradients(zip(grads, hyper_model.trainable_weights + base_model.trainable_weights[:-2]))

        
        Total_loss_tracker.update_state(loss_tracker.result())
        pbar.set_description("Epoch : %d / %d , Training Loss %f , Loss %f , Weigth STD %f , Weight Mean %f" % \
                             (epoch + 1, Epochs, float(Total_loss_tracker.result()), float(loss_tracker.result()), float(tf.math.reduce_std(generated_parameters)), float(tf.math.reduce_mean(generated_parameters))))
    pbar.close()
    loss_tracker.reset_states()

base_model.layers[-5].kernel = kernel1
base_model.layers[-4].kernel = kernel2


## Save Models
os.makedirs("weights", exist_ok=True)
hyper_model.save_weights("weights/hyper_model_weights.h5")
base_model.save_weights("weights/base_model_weights.h5")



## Test 
loss_tracker = tf.keras.metrics.Mean()

for test_channel in test_channels:
	test_channel = SVD(test_channel)

    Channel = np.stack([np.real(test_channel), np.imag(test_channel)], axis=-1)[None, ...]
    X = np.concatenate([np.real(samples), np.imag(samples)], axis=-1)

    real_y = generate_real_target(samples, test_channel)
    Y = np.concatenate([np.real(real_y), np.imag(real_y)], axis=-1)

    generated_parameters = hyper_model(Channel)
    parameterize_model(base_model, generated_parameters)

    preds = base_model(X)


    loss = loss_fn(Y, preds)
    loss_tracker.update_state(loss)
print(f"Test Loss : {float(loss_tracker.result())}")