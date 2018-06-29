# wincnn

A simple python module for computing minimal Winograd convolution algorithms for use with
convolutional neural networks as proposed in [1].

Requirements

+ python: version 2.7.6
+ sympy: version 1.0 (0.7.4.1 does not work)

## Example: F(2,3)

For F(m,r) you must select m+r-2 polynomial interpolation points.

In this example we compute transforms for F(2,3) or
F(2x2,3x3) using polynomial interpolation points (0,1,-1).

```
andrew@broadwell:~/develop/wincnn$ python
Python 2.7.11+ (default, Apr 17 2016, 14:00:29) 
[GCC 5.3.1 20160413] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import wincnn
>>> wincnn.showCookToomFilter((0,1,-1), 2, 3)
AT = 
⎡1  1  1   0⎤
⎢           ⎥
⎣0  1  -1  -1⎦

G = 
⎡ 1    0     0 ⎤
⎢              ⎥
⎢1/2  1/2   1/2⎥
⎢              ⎥
⎢1/2  -1/2  1/2⎥
⎢              ⎥
⎣ 0    0     1 ⎦

BT = 
⎡1  0   -1  0⎤
⎢            ⎥
⎢0  1   1   0⎥
⎢            ⎥
⎢0  -1  1   0⎥
⎢            ⎥
⎣0  1  0   -1⎦

AT*((G*g)(BT*d)) =
⎡d[0]⋅g[0] + d[1]⋅g[1] + d[2]⋅g[2]⎤
⎢                                 ⎥
⎣d[1]⋅g[0] + d[2]⋅g[1] + d[3]⋅g[2]⎦

```

The last matrix is the 1D convolution F(2,3) computed using the
transforms AT, G, and BT, on 4 element signal d[0..3] and 3 element
filter g[0..2], and serves to verify the correctness of the
transforms. This is a symbolic computation, so the result should be
exact.

## Example: F(4,3)

The following example computes transforms for F(4,3).

```
>>> wincnn.showCookToomFilter((0,1,-1,2,-2), 4, 3)
AT = 
⎡1  1  1   1  1   0⎤
⎢                  ⎥
⎢0  1  -1  2  -2  0⎥
⎢                  ⎥
⎢0  1  1   4  4   0⎥
⎢                  ⎥
⎣0  1  -1  8  -8  1⎦

G = 
⎡1/4     0     0  ⎤
⎢                 ⎥
⎢-1/6  -1/6   -1/6⎥
⎢                 ⎥
⎢-1/6   1/6   -1/6⎥
⎢                 ⎥
⎢1/24  1/12   1/6 ⎥
⎢                 ⎥
⎢1/24  -1/12  1/6 ⎥
⎢                 ⎥
⎣ 0      0     1  ⎦

BT = 
⎡4  0   -5  0   1  0⎤
⎢                   ⎥
⎢0  -4  -4  1   1  0⎥
⎢                   ⎥
⎢0  4   -4  -1  1  0⎥
⎢                   ⎥
⎢0  -2  -1  2   1  0⎥
⎢                   ⎥
⎢0  2   -1  -2  1  0⎥
⎢                   ⎥
⎣0  4   0   -5  0  1⎦

AT*((G*g)(BT*d)) =
⎡d[0]⋅g[0] + d[1]⋅g[1] + d[2]⋅g[2]⎤
⎢                                 ⎥
⎢d[1]⋅g[0] + d[2]⋅g[1] + d[3]⋅g[2]⎥
⎢                                 ⎥
⎢d[2]⋅g[0] + d[3]⋅g[1] + d[4]⋅g[2]⎥
⎢                                 ⎥
⎣d[3]⋅g[0] + d[4]⋅g[1] + d[5]⋅g[2]⎦
```
## Linear Convolution

If instead of an FIR filter you want the algorithm for linear convolution, all you have to do is exchange and transpose the data and inverse transform matrices. This is referred to as the Transfomation Principle.

```
>>> wincnn.showCookToomConvolution((0,1,-1),2,3)
A = 
⎡1  0 ⎤
⎢     ⎥
⎢1  1 ⎥
⎢     ⎥
⎢1  -1⎥
⎢     ⎥
⎣0  1 ⎦

G = 
⎡ 1    0     0 ⎤
⎢              ⎥
⎢1/2  1/2   1/2⎥
⎢              ⎥
⎢1/2  -1/2  1/2⎥
⎢              ⎥
⎣ 0    0     1 ⎦

B = 
⎡1   0  0   0 ⎤
⎢             ⎥
⎢0   1  -1  -1⎥
⎢             ⎥
⎢-1  1  1   0 ⎥
⎢             ⎥
⎣0   0  0   1 ⎦

Linear Convolution: B*((G*g)(A*d)) =
⎡      d[0]⋅g[0]      ⎤
⎢                     ⎥
⎢d[0]⋅g[1] + d[1]⋅g[0]⎥
⎢                     ⎥
⎢d[0]⋅g[2] + d[1]⋅g[1]⎥
⎢                     ⎥
⎣      d[1]⋅g[2]      ⎦
```

## Example: F(6,3)

This example computes transform for F(6,3). We will use fraction interpolation points 1/2
and -1/2, so we use sympy.Rational in order to keep the symbolic computation exact (using floating point values would make the derivation of the transforms subject to rounding error).

```
>>> from sympy import Rational
>>> wincnn.showCookToomFilter((0,1,-1,2,-2,Rational(1,2),-Rational(1,2)), 6, 3)
AT = 
⎡1  1  1   1    1    1      1    0⎤
⎢                                 ⎥
⎢0  1  -1  2   -2   1/2   -1/2   0⎥
⎢                                 ⎥
⎢0  1  1   4    4   1/4    1/4   0⎥
⎢                                 ⎥
⎢0  1  -1  8   -8   1/8   -1/8   0⎥
⎢                                 ⎥
⎢0  1  1   16  16   1/16  1/16   0⎥
⎢                                 ⎥
⎣0  1  -1  32  -32  1/32  -1/32  1⎦

G = 
⎡ 1      0     0  ⎤
⎢                 ⎥
⎢-2/9  -2/9   -2/9⎥
⎢                 ⎥
⎢-2/9   2/9   -2/9⎥
⎢                 ⎥
⎢1/90  1/45   2/45⎥
⎢                 ⎥
⎢1/90  -1/45  2/45⎥
⎢                 ⎥
⎢ 32    16        ⎥
⎢ ──    ──    8/45⎥
⎢ 45    45        ⎥
⎢                 ⎥
⎢ 32   -16        ⎥
⎢ ──   ────   8/45⎥
⎢ 45    45        ⎥
⎢                 ⎥
⎣ 0      0     1  ⎦

BT = 
⎡1   0    -21/4    0    21/4     0    -1  0⎤
⎢                                          ⎥
⎢0   1      1    -17/4  -17/4    1    1   0⎥
⎢                                          ⎥
⎢0   -1     1    17/4   -17/4   -1    1   0⎥
⎢                                          ⎥
⎢0  1/2    1/4   -5/2   -5/4     2    1   0⎥
⎢                                          ⎥
⎢0  -1/2   1/4    5/2   -5/4    -2    1   0⎥
⎢                                          ⎥
⎢0   2      4    -5/2    -5     1/2   1   0⎥
⎢                                          ⎥
⎢0   -2     4     5/2    -5    -1/2   1   0⎥
⎢                                          ⎥
⎣0   -1     0    21/4     0    -21/4  0   1⎦

AT*((G*g)(BT*d)) =
⎡d[0]⋅g[0] + d[1]⋅g[1] + d[2]⋅g[2]⎤
⎢                                 ⎥
⎢d[1]⋅g[0] + d[2]⋅g[1] + d[3]⋅g[2]⎥
⎢                                 ⎥
⎢d[2]⋅g[0] + d[3]⋅g[1] + d[4]⋅g[2]⎥
⎢                                 ⎥
⎢d[3]⋅g[0] + d[4]⋅g[1] + d[5]⋅g[2]⎥
⎢                                 ⎥
⎢d[4]⋅g[0] + d[5]⋅g[1] + d[6]⋅g[2]⎥
⎢                                 ⎥
⎣d[5]⋅g[0] + d[6]⋅g[1] + d[7]⋅g[2]⎦
```

[1] "Fast Algorithms for Convolutional Neural Networks" Lavin and Gray, CVPR 2016.
http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Lavin_Fast_Algorithms_for_CVPR_2016_paper.pdf
