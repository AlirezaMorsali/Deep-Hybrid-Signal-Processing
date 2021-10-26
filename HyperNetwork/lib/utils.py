import math
import numpy as np
import tensorflow as tf
from pyphysim.util.misc import randn_c
from tensorflow import keras
from tensorflow.keras import layers
from pyphysim.modulators.fundamental import BPSK, QAM, Modulator
from pyphysim.util.misc import randn_c
from pyphysim.util.conversion import dB2Linear, linear2dB
import numpy as np
from lib import hdnn


def mu_mm_ch_train(N, K, L, NN):
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


def get_Rx_ser(par,  model_rx):
    par.SNR_db = np.arange(par.SNR_range[0], par.SNR_range[1],
                           par.SNR_range[2])
    par.SNR = dB2Linear(par.SNR_db)
    par.modulator = BPSK() if par.MO == 2 else QAM(par.MO)
    test_power = np.array([])
    transmit_power_fd = np.array([])
    modulator = par.modulator
    SERFD = np.zeros(len(par.SNR))
    SERNN = np.zeros(len(par.SNR))
    NTS = par.Ns

    for ind in range(len(par.SNR)):
        P = par.SNR[ind]  # assuming sigma = 1
        TotbiterFD = 0
        TotbiterNN = 0
        TotpackFD = 0
        TotTrSym = 0

        while ((par.Minbiter > TotbiterFD) & ((TotTrSym < 10000) |
                                            (TotbiterFD < 1))):
        # while (par.Minbiter > TotbiterNN):

            # (HH, HC) = utils.mu_mm_ch_train(par.Ntx, par.Nrx, par.L, 1)
            (HH, HC) = mu_mm_ch_train(par.Ntx, par.Nrx, par.L, 1)
            H = HH.squeeze()
            # HT = tf.convert_to_tensor(HC.squeeze().T, dtype=tf.float32)
            U, D, V = np.linalg.svd(H, full_matrices=False)
            model_rx = hdnn.train_rx_model(par.Ns, par.Nrx,
                                           U, D, par.trs,
                                           P, par.BATCH_SIZE, par.EPOCHS,
                                           model_rx)
            PC = V.conj().T
            CB = (np.reshape(1 / D, [-1, 1]) * U.conj().T)
            data = np.random.randint(0, par.modulator.M,
                                     size=(par.paklen, par.Ns))
            sym = par.modulator.modulate(data)
            No = np.sqrt(1.0 / P) * randn_c(par.Nrx, par.paklen)
            # No = 0 for Testing
            XT = PC @ sym.T
            sig = np.linalg.norm(XT, ord=2, axis=0).reshape(
                [-1, 1]).repeat(par.Ntx, axis=1)
            XTt = XT / np.sqrt(sig.T)
            # XTt = PC @ sym[:NTS]
            xr_fd = (H @ XTt)
            xd_fd = xr_fd + No

            xd = np.concatenate((xd_fd.real, xd_fd.imag))
            # ad = xd.reshape([1, 2 * par.Nrx])
            ad = xd.T
            # Xr = model_rx.predict(ad)[0]
            Xr = model_rx.predict(ad)
            Xdet = Xr[:, :par.Ns] + 1j * Xr[:, par.Ns:]
            ddata_nn = modulator.demodulate(Xdet)
            TotbiterNN += sum(sum(ddata_nn != data))

            DsymFD = CB @ xd_fd
            ddata_fd = modulator.demodulate(DsymFD)
            TotbiterFD += sum(sum(ddata_fd.T != data))
            TotTrSym += NTS*par.paklen
            TotpackFD += 1
            power = XTt.conj().T @ XTt
            transmit_power_fd = np.append(transmit_power_fd, power)

            power = xr_fd.conj().T @ xr_fd
            test_power = np.append(test_power, power)
            (print('\nSNR = ', par.SNR_db[ind], 'Remained Realization = ',
                ((par.Minbiter - TotbiterFD) >= 0) *
                (par.Minbiter - TotbiterFD), '\n'))
            # print(
            #     'SNR ={}, Total sent symbols={}, Symbol error rate={:.2f}'.format(par.SNR_db[ind], TotTrSym, TotbiterNN / (TotTrSym)))
        SERFD[ind] = TotbiterFD / (TotTrSym)
        SERNN[ind] = TotbiterNN / (TotTrSym)
    return (SERFD, SERNN)


def get_tx_ser(par, model_tx):
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
            (HH, HC) = mu_mm_ch_train(par.Ntx, par.Nrx, par.L, 1)
            H = HH.squeeze()
            uu, ss, vv = np.linalg.svd(H, full_matrices=False)
            CB = (1 / ss.reshape([-1, 1]) * uu.T.conj())
            PC = (vv.T.conj() / np.linalg.norm(vv))
            pc_u = np.concatenate((PC.real, -PC.imag), axis=1)
            pc_d = np.concatenate((PC.imag, PC.real), axis=1)
            pc = np.concatenate((pc_u, pc_d), axis=0)
            model_tx = hdnn.train_tx_model(x_hat_i, pc, par.BATCH_SIZE, par.EPOCHS,
                                           model_tx)
            # cntframe += 1
            data = np.random.randint(0, modulator.M, size=(par.paklen,
                                                            par.Ns))
            sym = modulator.modulate(data)
            xr = sym.real
            xi = sym.imag
            xcon = np.concatenate((xr, xi), axis=1)
            Xr = model_tx.predict(xcon)
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
