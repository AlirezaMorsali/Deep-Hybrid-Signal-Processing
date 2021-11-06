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
optimizer = tf.keras.optimizers.Adam(2e-4)
loss_fn = tf.keras.losses.MeanSquaredError()

First_Weights = base_model.get_weights()


Epochs = 1
batch_size = 4 ** 8
loss_tracker = tf.keras.metrics.Mean()
Total_loss_tracker = tf.keras.metrics.Mean()


for epoch in range(Epochs):

    X = np.concatenate([np.real(samples), np.imag(samples)], axis=1)

    flag = 1
    pbar = tqdm(train_channels)
    for train_channel in pbar:

        train_channel = SVD(train_channel)

        real_y = generate_real_target(samples, train_channel)
        Y = np.concatenate([np.real(real_y), np.imag(real_y)], axis=1)
        Channel = np.stack([np.real(train_channel), np.imag(train_channel), np.angle(train_channel), np.abs(train_channel)], axis=2)[None, ...]

        for counter in range(int(4 ** 8 / batch_size)):
            loss_tracker.reset_states()
            x_batch, y_batch =  X[counter * batch_size:(counter + 1) * batch_size], Y[counter * batch_size:(counter + 1) * batch_size]
            with tf.GradientTape() as tape:
                generated_parameters = hyper_model(Channel)
                parameterize_model(base_model, generated_parameters)

                preds = base_model(x_batch)
                loss = loss_fn(y_batch, preds)
                loss_tracker.update_state(loss)
            grads = tape.gradient(loss, hyper_model.trainable_weights + base_model.trainable_weights)
            optimizer.apply_gradients(zip(grads, hyper_model.trainable_weights + base_model.trainable_weights))

        
        Total_loss_tracker.update_state(loss_tracker.result())
        pbar.set_description("Epoch : %d / %d , Training Loss %f , Loss %f , Learning Rate %f" % \
                             (epoch + 1, Epochs, float(Total_loss_tracker.result()), float(loss_tracker.result()), optimizer.learning_rate.numpy()))
        
        if flag % 500 == 0:
            optimizer.lr.assign(optimizer.learning_rate.numpy() * np.exp(-0.15))
        flag += 1
    pbar.close()
    loss_tracker.reset_states()


Last_Weights = base_model.get_weights()
flag = 0
for counter, weight in enumerate(First_Weights):
    if First_Weights[counter].shape == Last_Weights[flag].shape:
        First_Weights[counter] = Last_Weights[flag]
        flag += 1
base_model = base_network()
base_model.set_weights(First_Weights)


## Save Models
os.makedirs("weights", exist_ok=True)
hyper_model.save_weights("weights/hyper_model_weights.h5")
base_model.save_weights("weights/base_model_weights.h5")



## Test 
loss_tracker = tf.keras.metrics.Mean()

for counter, test_channel in enumerate(test_channels):
    if counter % 100 == 0:
        print(f"Counter = {counter}")
    test_channel = SVD(test_channel)

    Channel = np.stack([np.real(test_channel), np.imag(test_channel), np.angle(train_channel), np.abs(train_channel)], axis=2)[None, ...]
    X = np.concatenate([np.real(samples), np.imag(samples)], axis=1)

    real_y = generate_real_target(samples, test_channel)
    Y = np.concatenate([np.real(real_y), np.imag(real_y)], axis=1)

    generated_parameters = hyper_model(Channel)
    parameterize_model(base_model, generated_parameters)

    preds = base_model(X)


    loss = loss_fn(Y, preds)
    loss_tracker.update_state(loss)
print(f"Test Loss : {float(loss_tracker.result())}")