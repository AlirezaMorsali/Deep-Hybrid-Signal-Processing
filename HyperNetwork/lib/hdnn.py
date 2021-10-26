import numpy as np
import tensorflow as tf
from pyphysim.util.misc import randn_c
from tensorflow import keras
from tensorflow.keras import layers


def get_rx_model(Ns, Nrx, n_rf, nd_layer=5, na_layer=1, wid_d=3, wid_a=3):
    '''
     HDNN model for Uplink connention Nrf>= Ns
    '''
    input_complex = keras.Input(shape=(2 * Nrx))
    s = layers.Dense(wid_a * Nrx, activation="relu")(input_complex)
    z = layers.Dense(wid_a * Nrx, activation="relu")(input_complex)
    for _ in range(na_layer-1):  # Analog DNN
        s = layers.Dense(wid_a * Nrx, activation="relu")(s)
        z = layers.Dense(wid_a * Nrx, activation="relu")(z)
    s = layers.Dense(n_rf)(s)  # RF chains
    z = layers.Dense(n_rf)(z)
    for _ in range(nd_layer):  # Digital DNN
        s = layers.Dense(wid_d * Ns, activation="relu")(s)
        z = layers.Dense(wid_d * Ns, activation="relu")(z)
    s = layers.Dense(Ns)(s)
    z = layers.Dense(Ns)(z)
    dec_signal = layers.concatenate([s, z], axis=1)

    return keras.Model(inputs=input_complex,
                       outputs=dec_signal,
                       name="RxBuild")


def get_tx_model(Ns, Ntx, n_rf, nd_layer=5, na_layer=1, wid_d=2, wid_a=3):
    '''
     HDNN model for Downling connention
    '''
    input_complex = keras.Input(shape=(2 * Ns))
    s = layers.Dense(wid_d * Ns, activation="relu")(input_complex)
    z = layers.Dense(wid_d * Ns, activation="relu")(input_complex)
    for _ in range(nd_layer-1):  # Analog DNN
        s = layers.Dense(wid_d * Ns, activation="relu")(s)
        z = layers.Dense(wid_d * Ns, activation="relu")(z)
    s = layers.Dense(n_rf)(s)  # RF chains
    z = layers.Dense(n_rf)(z)
    for _ in range(na_layer):  # Digital DNN
        s = layers.Dense(wid_a * Ntx, activation="relu")(s)
        z = layers.Dense(wid_a * Ntx, activation="relu")(z)
    s = layers.Dense(Ntx)(s)
    z = layers.Dense(Ntx)(z)
    trx_signal = layers.concatenate([s, z], axis=1)

    return keras.Model(inputs=input_complex,
                       outputs=trx_signal,
                       name="TxBuild")


def train_rx_model(Ns, Nrx, U, D, trs, Pow, BATCH_SIZE,
                   EPOCHS, model_rx):
    CB = (np.reshape(1 / D, [-1, 1]) * U.conj().T)
    noise = np.sqrt(1.0 / Pow) * randn_c(trs, Nrx)
    Sran = randn_c(trs, Nrx)
    yran = np.matmul(Sran+noise, CB.T)
    yran_i = np.concatenate((yran.real, yran.imag), axis=1)
    xran = Sran + noise
    xran_i = np.concatenate((xran.real, xran.imag), axis=1)
    xran_ten = tf.convert_to_tensor(xran_i, dtype=tf.float32)
    yran_ten = tf.convert_to_tensor(yran_i, dtype=tf.float32)
    x_train = tf.data.Dataset.from_tensor_slices(
        (xran_ten, yran_ten)).batch(BATCH_SIZE)
    model_rx.fit(x_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
    return model_rx

def train_tx_model(x_hat_i, pc, BATCH_SIZE, EPOCHS, model_tx):
    yt = np.matmul(x_hat_i, pc.T)
    y_true = tf.convert_to_tensor(yt, dtype=tf.float32)
    x_train = tf.data.Dataset.from_tensor_slices(
        (x_hat_i, y_true)).batch(BATCH_SIZE)
    model_tx.fit(x_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
    return model_tx
