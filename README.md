# Image-Reconstruction using convolutional auto-encoder (CAE)
Dataset used: MNIST dataset, comprising of 60000 examples as training set and 10000 examples as testing set. 

# Model Description
Convolutional Auto-encoder (CAE) is based on the encoder-decoder model that works by transforming the input into a lower-dimensional representation (encoder part) and then reconstructing the output from this representation (decoder part).

The implemented encoder consists of convolutional layers and max-pooling layers. The max-pooling layer introduces sparsity over the hidden representation by erasing all non-maximal values in non-overlapping sub-regions. The combination of convolution and pooling provides both translation invariance feature detection and a compact image representation for the decoder. Activation function ReLU is used after each convolution layer to introduce non-linearity to the network.
The input image is converted from wide and thin [28 x 28 pixels with 1 channel] to narrow and thick [4 x 4 pixels with 64 channels (feature maps)] as an output of encoder which is then fed to decoder.

The decoder needs to convert from a narrow representation to a wide reconstructed image. Transposed convolution layers are used to increase the width and height of the layers. They work almost exactly the same as convolutional layers, but in reverse. The use of hyperbolic tangent activation function, which constrain the resulting values of feature maps to the interval [-1,1], sets appropriate limits on the values of feature maps at the end of the decoder, and provides good convergence of the whole model. The decoder outputs deconvolved image [28 x 28 pixels with 1 channel].
