# Disentangling hidden factors of variations in Deep Networks
Course project for Machine learning for Signal Processing ([MLSP](http://leap.ee.iisc.ac.in/sriram/teaching/MLSP_19/)) course: E9 205, at the Indian Institute of Science ([IISc](https://iisc.ac.in/)).

The work is based on the [paper](https://arxiv.org/abs/1412.6583).
We trained an Autoencoder to discover the hidden factors of variations such as subject identity for the expression classification task and qualitatively measured the accuracy by swapping the face expressions across the subject identity.
For more details about the project, please refer to the [Project Report](https://github.com/chaitrasj/Disentangling-hidden-factors-of-variations-in-Deep-Networks/blob/main/Disentangling_hidden_factors_of_variations_in_Deep_Networks.pdf).

### Datasets
- MNIST
- [JAFFE](https://paperswithcode.com/dataset/jaffe#:~:text=The%20JAFFE%20dataset%20consists%20of,facial%20expression%20by%2060%20annotators.) (Japanese Female Facial Expression)

### Training
Training code is available at [Jaffe.ipynb](https://github.com/chaitrasj/Disentangling-hidden-factors-of-variations-in-Deep-Networks/blob/main/JAFEE.ipynb) for JAFFE dataset and at [Mnist.ipynb](https://github.com/chaitrasj/Disentangling-hidden-factors-of-variations-in-Deep-Networks/blob/main/MNIST.ipynb) for MNIST dataset.

For more details about the Training, testing procedures and the losses used, please refer to the [Project Report](https://github.com/chaitrasj/Disentangling-hidden-factors-of-variations-in-Deep-Networks/blob/main/Disentangling_hidden_factors_of_variations_in_Deep_Networks.pdf).
