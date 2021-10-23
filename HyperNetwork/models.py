import numpy as np
import tensorflow as tf



def parameterize_model(model, weights):
    flag = 0
    for layer in model.layers[-5:]:
        if 'dense' in layer.name: 
              new_weights = weights[0, :, :, flag]
              layer.kernel = new_weights
              flag += 1


def complex_dense(x, units, in_features, AF=False):
    RL = tf.keras.layers.Dense(units, use_bias=False)
    IL = tf.keras.layers.Dense(units, use_bias=False) 

    xr = tf.keras.layers.Lambda(lambda x: x[:,:in_features // 2])(x)
    xi = tf.keras.layers.Lambda(lambda x: x[:,in_features // 2:])(x)


    xr_1 = RL(xr)
    xr_2 = IL(xi)
    Xr = tf.keras.layers.Subtract()([xr_1, xr_2])
    if AF:
      #Xr = tf.keras.layers.ReLU()(Xr)
      Xr = tf.keras.layers.Lambda(lambda x: tf.nn.tanh(x))(Xr)

    xi_1 = RL(xi)
    xi_2 = IL(xr)
    Xi = tf.keras.layers.Add()([xi_1, xi_2])
    if AF:
      #Xi = tf.keras.layers.ReLU()(Xi)
      Xi = tf.keras.layers.Lambda(lambda x: tf.nn.tanh(x))(Xi)


    y = tf.keras.layers.Concatenate()([Xr,Xi])

    return y



def base_network():
    Input = tf.keras.layers.Input(shape=(2 * 8))

    x = complex_dense(Input, 32, 2 * 8, True)
    x = complex_dense(x, 32, 2 * 32, True)
    x = complex_dense(x, 32, 2 * 32, True)
    x = complex_dense(x, 8, 2 * 32, True)

    y = complex_dense(x, 128, 2 * 8)

    base_model = tf.keras.models.Model(Input, y)

    #base_model.summary()

    return base_model



def hyper_network():
    hyper_model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(8, 128, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), padding="same"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, (3, 3), padding="same"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(128, (3, 3), padding="same"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(2, (3, 3), padding="same"),
        ], name='hyper_model'
    )

    #hyper_model.summary()
    
    return hyper_model



def load_model(model_fn, path_weights):
	model = model_fn()
	model.load_weights(path_weights)

	return model