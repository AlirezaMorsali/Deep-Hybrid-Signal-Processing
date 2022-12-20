# Deep Learning-Based Hybrid Analog-Digital Signal Processing

This is the Tensorflow 2.x implementation of our paper ["Deep Learning-Based Hybrid Analog-Digital Signal Processing in mmWave Massive-MIMO Systems"](https://ieeexplore.ieee.org/abstract/document/9815247), IEEE Access, vol. 10, pp. 72348-72362, 2022. 

Hybrid analog-digital signal processing (HSP) is an enabling technology to harvest the potential of millimeter-wave (mmWave) massive-MIMO communications. In this paper, we present a general deep learning (DL) framework for efficient design and implementation of HSP-based massive-MIMO systems. Exploiting the fact that any complex matrix can be written as a scaled sum of two matrices with unit-modulus entries, a novel analog deep neural network (ADNN) structure is first developed which can be implemented with common radio frequency (RF) components. This structure is then embedded into an extended hybrid analog-digital deep neural network (HDNN) architecture which facilitates the implementation of mmWave massive-MIMO systems while improving their performance. In particular, the proposed HDNN architecture enables HSP-based massive-MIMO transceivers to approximate any desired transmitter and receiver mapping with arbitrary precision. To demonstrate the capabilities of the proposed DL framework, we present a new HDNN-based beamformer design that can achieve the same performance as fully-digital beamforming, with reduced number of RF chains. Finally, simulation results are presented confirming the advantages of the proposed HDNN design over existing hybrid beamforming schemes.


## Citation

If you find our code useful for your research, please consider citing:
```bibtex
@article{morsali2022deep,
  title={Deep Learning-Based Hybrid Analog-Digital Signal Processing in mmWave Massive-MIMO Systems},
  author={Morsali, Alireza and Haghighat, Afshin and Champagne, Benoit},
  journal={IEEE Access},
  volume={10},
  pages={72348--72362},
  year={2022},
  publisher={IEEE}
}
```
