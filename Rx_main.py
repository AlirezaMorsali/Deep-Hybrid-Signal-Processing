import argparse
from matplotlib import pyplot as plt
from tensorflow import keras
from lib import hdnn, utils
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser(description='Uplink Hybrid Massive MIMO')
parser.add_argument('-L', '--Lpath', dest='L', type=int,  default=20,
                    help='Number of paths')
parser.add_argument('-Ntx', '--NumTranAnt', dest='Ntx', type=int,  default=8,
                    help='Number of transmit antennas')
parser.add_argument('-Nrx', '--NumRecvAnt', type=int, dest='Nrx', default=128,
                    help='Number of receive antennas')
parser.add_argument('-Ns', '--Nsyms', dest='Ns', type=int,  default=8,
                    help='Number of symbols per transmission')
parser.add_argument('-Nrf', '--NRFchaisn', dest='Nrf', type=int,  default=8,
                    help='Number of RF chains')
parser.add_argument('-Mo', '--ModulationOrder', dest='MO', type=int, default=4,
                    choices=[2, 4, 16, 64], help='Number of RF chains')
parser.add_argument('-Mber', '--BinBitErr', dest='Minbiter', type=int,
                    default=1000, help='Minimum number of erros per SNR')
parser.add_argument('-S', '--SNR_rnage', dest='SNR_range', nargs=3, type=int,
                    default=[-10, 0, 2],
                    help='SNR range [-10, 0, 2]-> np.arange(-10, 0, 2)')
parser.add_argument('-trs', '--trainingsize', dest='trs',
                    type=int, default=500000,
                    help='Number of symbol vectors for training')
parser.add_argument('-lr', '--LearningRage', dest='learning_rate', type=float,
                    default=0.001, help='Learning rate')
parser.add_argument('-fn', '--filename', dest='version_name', default="Uplink",
                    help='Prefix name of the generated files')
parser.add_argument('-bs', '--BatchSize', dest='BATCH_SIZE', type=int,
                    default=50, help='Batch Size')
parser.add_argument('-ep', '--Epochs', dest='EPOCHS', type=int,
                    default=5, help='Number of epochs')
parser.add_argument('-fl', '--FrameLenght', dest='paklen', type=int,
                    default=500,
                    help='Length of frame per channel realizaiton')


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


if is_interactive():
    par = parser.parse_args([])
else:
    par = parser.parse_args()

# Creating and compiling the model
model_rx = hdnn.get_rx_model(par.Ns, par.Nrx, par.Ns)
model_rx.summary()
keras.utils.plot_model(model_rx, par.version_name + "_Model.png",
                       show_shapes=True)
model_rx.compile(
    # optimizer=keras.optimizers.Adam(learning_rate=par.learning_rate),
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.MeanAbsoluteError())

# Symbol error rate
(SERFD, SERNN) = utils.get_Rx_ser(par, model_rx)


fig, ax = plt.subplots(figsize=(10, 10))
ax.semilogy(par.SNR_db, SERNN, "--", label="NN")
ax.semilogy(par.SNR_db, SERFD, "-.", label="FD")
ax.set_title("Ns = {}, N_rf={}, Nt = {}, L ={}, Nr = {}".format(
    par.Ns, par.Nrf, par.Ntx, par.L, par.Nrx))
ax.set_ylabel("Symbol Error Rate")
ax.set_xlabel("SNR (dB)")
ax.legend()
fig.savefig(par.version_name + '_SER.png')
