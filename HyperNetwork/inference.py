import numpy as np
import tensorflow as tf

from utils import *
from models import *


def inference(x, channel, base_model, hyper_model):
'''
This fucntion is made for inference time and is the main function.
inputs:
    x : input( complex and shape (batch_size, N) )
    channel : channel not PC( complex and shape (N, n_receiver))
    base_model : pretrained base model
    hyper_model : pretrained hyper model
output:
    pred : concatenated output(batch_size, [real, imag])
'''
    channel = SVD(channel)
    channel = np.stack([np.real(channel), np.imag(channel)], axis=-1)[None, ...]
    x = np.concatenate([np.real(x), np.imag(x)], axis=-1)

    generated_parameters = hyper_model(channel)
    parameterize_model(base_model, generated_parameters)
    pred = base_model(x)

    return pred
