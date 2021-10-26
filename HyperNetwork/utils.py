import numpy as np
from pyphysim.util.misc import randn_c
from pyphysim.modulators.fundamental import BPSK, QAM, Modulator
from pyphysim.util.conversion import dB2Linear, linear2dB

from models import *


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

def SVD(H):
    U, D, V = np.linalg.svd(H, full_matrices=False)
    PC = V.conj()
    return PC


def generate_real_target(samples, PC):
    XT = PC.T @ samples.T

    return XT.T


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

def get_tx_ser(par, base_model, hyper_model):
    par.SNR_db = np.arange(par.SNR_range[0], par.SNR_range[1],
                           par.SNR_range[2])
    par.SNR = dB2Linear(par.SNR_db)
    modulator = BPSK() if par.MO == 2 else QAM(par.MO)
    SERFD = np.zeros(len(par.SNR))
    SERNN = np.zeros(len(par.SNR))
    NTS = par.Ns
    # model = model_tx
    data_i = np.random.randint(0, modulator.M, size=[par.Ns, par.trs])
    sym_i = modulator.modulate(data_i).T
    x_hat_i = np.concatenate((sym_i.real, sym_i.imag), axis=1)

    for ind in range(len(par.SNR)):
        P = par.SNR[ind]  # assuming sigma = 1
        TotbiterFD = 0
        TotbiterNN = 0
        TotpackFD = 0
        TotTrSym = 0
        ind2 = 1

        while ((par.Minbiter > TotbiterFD) & ((TotTrSym < 10000) |
                                            (TotbiterFD < 1))):
            HH = mu_mm_ch_train(par.Ntx, par.Nrx, par.L, 1)
            H = HH.squeeze()
            uu, ss, vv = np.linalg.svd(H, full_matrices=False)
            CB = (1 / ss.reshape([-1, 1]) * uu.T.conj())
            PC = vv.T.conj()
            # PC = (vv.T.conj() / np.linalg.norm(vv))
            # TODO
            # Train with normalized pc
            data = np.random.randint(0, modulator.M, size=(par.paklen,
                                                            par.Ns))
            sym = modulator.modulate(data)
            Xr = inference(sym, H, base_model, hyper_model).numpy()
            # Xr = model_tx.predict(xcon)
            No = np.sqrt(1.0 / P) * randn_c(par.Nrx, par.paklen)
            X = Xr[:, :par.Ntx] + 1j * Xr[:, par.Ntx:]
            sig = np.linalg.norm(X, ord=2, axis=1).reshape(
                [-1, 1]).repeat(par.Ntx, axis=1)
            xt_nn = X / np.sqrt(sig)
            # No = 0
            xr_nn = (H @ xt_nn.T)
            xd_nn = xr_nn + No
            ddata_nn = modulator.demodulate(CB@xd_nn)
            TotbiterNN += sum(sum(ddata_nn != data.T))

            XT = (PC @ sym.T).T
            sig = np.linalg.norm(XT, ord=2, axis=1).reshape(
                [-1, 1]).repeat(par.Ntx, axis=1)
            XTt = XT / np.sqrt(sig)
            # XTt = PC @ sym[:NTS]
            xr_fd = (H @ XTt.T)
            xd_fd = xr_fd + No
            DsymFD = CB@xd_fd
            ddata_fd = modulator.demodulate(DsymFD)
            TotbiterFD += sum(sum(ddata_fd != data.T))
            TotTrSym += NTS*par.paklen
            TotpackFD += 1
            (print('\nSNR = ', par.SNR_db[ind], 'Remained Realization = ',
                ((par.Minbiter - TotbiterFD) >= 0) *
                (par.Minbiter - TotbiterFD), '\n'))
            ind2 = ind2 + 1
        SERFD[ind] = TotbiterFD / (TotTrSym)
        SERNN[ind] = TotbiterNN / (TotTrSym)
    return (SERFD, SERNN)
