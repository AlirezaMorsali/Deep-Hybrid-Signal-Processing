import numpy as np
import tensorflow as tf

from pyphysim.util.misc import randn_c
from pyphysim.modulators.fundamental import BPSK, QAM, Modulator




def generate_sample(n_symbols=4, n_symbol_array=8):
    samples = []
    for array in range(n_symbols ** n_symbol_array):
        sample = np.zeros(n_symbol_array)
        for counter in range(n_symbol_array):
            sample[counter] = array % n_symbols
            array = array // n_symbols
            
        samples.append(sample)   
    
    samples = np.stack(samples).astype(np.int32)
    modulator = BPSK() if n_symbols == 2 else QAM(n_symbols)
    sym_samples = modulator.modulate(samples)

    return sym_samples



def mu_mm_ch_train(N, K, L=20, NN=1000):
    """
    This function generates NN relaization of KxN channel
    Inputs:
        N: Number of transmit antennas
        K: Number of receive antennas, or number of single antenna users
        L: Number of paths
        NN: Number of channel reallization
    output:
        H: (NN, K, N) complex-values numpy array
    """

    Phi_r = 2 * np.pi * np.random.uniform(0, 1, [NN, 1, K, L])
    T_r = np.cos(np.repeat(Phi_r, N, axis=1))
    kdr = -np.pi * 1j * (np.repeat(np.arange(N).reshape(1, N, 1, 1), L,
                                   axis=3))
    a_r = 1 / np.sqrt(N) * np.exp(T_r * (kdr))
    P = np.random.uniform(0, 1, [NN, 1, K, L])
    NF = np.repeat(np.sum(P, axis=3)[:, :, :, np.newaxis], L, axis=3)
    P = P / NF
    alpha1 = randn_c(NN, 1, K, L)
    alpha = np.repeat(alpha1 * np.sqrt(P), N, axis=1)
    H = np.transpose(
        np.sqrt(N) * np.sum(alpha * (a_r.squeeze()), axis=3), [0, 2, 1])
    
    return H



def generate_real_target(samples, PC):
    XT = PC.T @ samples.T

    return XT.T