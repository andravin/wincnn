# Wincnn FAQ

### Can Winograd fast convolution algorithms be used with strided convolutions?

It is possible to use Winograd or any convolution algorithm with strides > 1 by decimating the input and filter to make un-strided convolutions, then adding the results.

For example:
```
xoxoxoxoxox
*2
xox
=
xxxxxx * xx + ooooo * o
```

So a 1D stride 2 convolution can be decomposed into the sum of two un-strided convolutions, each using half of the data and filter elements. You can use a Winograd algorithm on each of the un-strided convolutions.

The same technique can be used with strided 2D convolutions, but then you need a sum of 4 un-strided convolutions.

Any book about fast digital signal processing algorithms will have a chapter on "Decimated Convolution" (ie strided convolution), but they usually only discuss the 1D case.

I probably first became aware of this technique from the following paper by Brosch and Tam, which does use 2D decimation in conjunction with FFT convolution: https://www.researchgate.net/profile/Tom_Brosch/publication/267930193_Efficient_Training_of_Convolutional_Deep_Belief_Networks_in_the_Frequency_Domain_for_Application_to_High-Resolution_2D_and_3D_Images/links/55f05f3d08ae0af8ee1d1904.pdf

### What about dilated convolutions?

In general, the Winograd algorithm or any fast convolution algorithm can be used with dilated convolution.

This is easily seen if we first consider that dilated convolution is really just convolution on a shifted, decimated input. That is, in order to compute a dilated convolution with scale 2<sup>i</sup>, you can first decimate the input signal X by removing all the rows and columns except for the ones at distance 2<sup>i</sup>, to give a decimated signal X<sub>i</sub>. You then perform regular convolution on X<sub>i</sub>.

You need to repeat this process for all shifts of the signal X by (j,k) pixels where (0,0) <= (j,k) < (2<sup>i</sup>, 2<sup>i</sup>). Call each of these decimated shifts X<sub>ijk</sub>, then the dilated convolution is the union of all the Y<sub>ijk</sub> = F * X<sub>ijk</sub>.

Now each of the F * X<sub>ijk</sub> is a regular convolution, so they each can be performed with a fast algorithm such as Winograd.
