#!/usr/bin/env jupyter

import math
import numpy as np
import numpy.matlib as mt
from pyphysim.util.misc import randn_c
from pyphysim.modulators.fundamental import BPSK, QAM, Modulator
from pyphysim.util.misc import randn_c
from pyphysim.util.conversion import dB2Linear, linear2dB


def mu_mm_ch_train(N=128, K=8, NN=1, L=20):
    """
    This function generates NN relaization of KxN channel
    Inputs:
        N: Number of transmit antennas
        K: Number of receive antennas, or number of single antenna users
        L: Number of paths
        NN: Number of channel reallization
    output:
        H: (NN, K, N) complex-values numpy array
        HC: (NN, 2K, 2N) real numpy array
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
    Hu = np.concatenate((H.real, -H.imag), axis=2)
    Hd = np.concatenate((H.imag, H.real), axis=2)
    HC = np.concatenate((Hu, Hd), axis=1)
    return (H, HC)


def allthestates(s, n):
    """
    Helper function
    This function return all the M^N possiple combination of elements of vector
    s with repetation
    O = allthestates(s) returns a M*N larg matrix O,
    M = length(s)^p : all the possible ways of combination of elements of vector s;
    """
    p = len(s)
    o = np.zeros((p**n, n))
    for i in range(1, n+1):
        t1 = np.repeat(s, p**(i-1))
        t2 = np.reshape(t1, (1, -1))
        t3 = mt.repmat(t2, 1, p**(n-i))
        o[:, n-i] = t3
    return o


def gen_all_syms(ns=8, mod_order=2):
    """
    This function generated all possible mod_order^ns symbol vectors
    Inputs:
    ns: number of symbols
    mod_order: modulation order
    Output:
    sym: mod_order^ns x ns matrix
    """
    modulator = BPSK() if mod_order == 2 else QAM(mod_order)
    bits = allthestates(np.arange(mod_order), ns)
    sym = modulator.modulate(bits.astype('int'))
    return sym


def gen_coded(sym, chnl_mtrx):
    """
    This function generates desired signal for given symbols and channel matrix
    Inputs:
    sym: MxNs sybmols where Ns is the number of symbols to be precoded and M is the number of packests
    chnl_mtrx: 1xNsxN channel matrix, wehre N is the number of transmit antennas and K is the number of receive antennas
    """
    H = chnl_mtrx.squeeze()
    U, D, V = np.linalg.svd(H, full_matrices=False)
    PC = V.conj().T
    XT = PC @ sym.T
    return XT

if __name__ == '__main__':
    chnl_mtrx,_ = mu_mm_ch_train()
    x = gen_all_syms()
    y = gen_coded(x, chnl_mtrx)
