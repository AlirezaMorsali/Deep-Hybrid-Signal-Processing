import numpy as np
import tensorflow as tf


def parameterize_model(model, weights):

    layer_count = 0
    sublayer_count = 0
    for layer in model.layers[-14:]:
        if 'dense' in layer.name: 
              new_weights = weights[layer_count][0, :, :, sublayer_count]
              layer.kernel = new_weights
              sublayer_count += 1

              if sublayer_count % 2 == 0:
                  layer_count += 1
                  sublayer_count = 0


def complex_dense(x, units, in_features, use_AF=False, AF=tf.nn.tanh):
    RL = tf.keras.layers.Dense(units, use_bias=True)
    IL = tf.keras.layers.Dense(units, use_bias=True) 

    xr = tf.keras.layers.Lambda(lambda x: x[:,:in_features // 2])(x)
    xi = tf.keras.layers.Lambda(lambda x: x[:,in_features // 2:])(x)


    xr_1 = RL(xr)
    xr_2 = IL(xi)
    Xr = tf.keras.layers.Subtract()([xr_1, xr_2])
    if use_AF:
      #Xr = tf.keras.layers.ReLU()(Xr)
      Xr = tf.keras.layers.Lambda(lambda x: AF(x))(Xr)

    xi_1 = RL(xi)
    xi_2 = IL(xr)
    Xi = tf.keras.layers.Add()([xi_1, xi_2])
    if use_AF:
      #Xi = tf.keras.layers.ReLU()(Xi)
      Xi = tf.keras.layers.Lambda(lambda x: AF(x))(Xi)


    y = tf.keras.layers.Concatenate()([Xr,Xi])

    return y



def base_network():
    Input = tf.keras.layers.Input(shape=(2 * 8))

    x = complex_dense(Input, 32, 2 * 8, True)
    x = complex_dense(x, 32, 2 * 32, True)
    x = complex_dense(x, 32, 2 * 32, True)
    x = complex_dense(x, 8, 2 * 32, True)

    x = complex_dense(x, 2048, 2 * 8, True, AF=tf.nn.relu)
    y = complex_dense(x, 128, 2 * 2048)

    base_model = tf.keras.models.Model(Input, y)

    #base_model.summary()

    return base_model



def hyper_network():
    Input = tf.keras.layers.Input(shape=(8, 128, 4))
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(Input)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same")(x)
    x = tf.keras.layers.ReLU()(x)
    x1 = tf.keras.layers.Conv2D(256, (3, 3), padding="same")(x)


    first_layer = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(1, 4), padding="same")(x1)
    first_layer = tf.keras.layers.ReLU()(first_layer)
    first_layer = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(1, 4), padding="same")(first_layer)
    first_layer = tf.keras.layers.ReLU()(first_layer)
    first_layer = tf.keras.layers.Conv2D(2, (3, 3), padding="same")(first_layer)



    second_layer = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(16, 1), padding="same")(x1)
    second_layer = tf.keras.layers.ReLU()(second_layer)
    second_layer = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(16, 1), padding="same")(second_layer)
    second_layer = tf.keras.layers.ReLU()(second_layer)
    second_layer = tf.keras.layers.Conv2D(2, (3, 3), padding="same")(second_layer)

    
    hyper_model = tf.keras.models.Model(Input, [first_layer, second_layer])
    
    #hyper_model.summary()
    
    return hyper_model



def load_model(model_fn, path_weights):
	model = model_fn()
	model.load_weights(path_weights)

	return model