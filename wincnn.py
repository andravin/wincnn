import operator
from functools import reduce
from typing import Sequence, Tuple

import math

from sympy import (
    Float,
    IndexedBase,
    Matrix,
    N,
    Poly,
    Rational,
    cos,
    pi,
    simplify,
    symbols,
    zeros,
    pprint,
)


def At(a, m, n):
    return Matrix(m, n, lambda i, j: a[i] ** j)


def A(a, m, n):
    return At(a, m - 1, n).row_insert(
        m - 1, Matrix(1, n, lambda i, j: 1 if j == n - 1 else 0)
    )


def T(a, n):
    return Matrix(
        Matrix.eye(n).col_insert(n, Matrix(n, 1, lambda i, j: -(a[i] ** n)))
    )


def Lx(a, n):
    x = symbols("x")
    return Matrix(
        n,
        1,
        lambda i, j: Poly(
            reduce(
                operator.mul,
                ((x - a[k] if k != i else 1) for k in range(0, n)),
                1,
            ).expand(basic=True),
            x,
        ).as_expr(),
    )


def F(a, n):
    return Matrix(
        n,
        1,
        lambda i, j: reduce(
            operator.mul,
            ((a[i] - a[k] if k != i else 1) for k in range(0, n)),
            1,
        ),
    )


def Fdiag(a, n):
    f = F(a, n)
    return Matrix(n, n, lambda i, j: (f[i, 0] if i == j else 0))


def FdiagPlus1(a, n):
    f = Fdiag(a, n - 1)
    f = f.col_insert(n - 1, zeros(n - 1, 1))
    f = f.row_insert(n - 1, Matrix(1, n, lambda i, j: (1 if j == n - 1 else 0)))
    return f


def L(a, n):
    x = symbols("x")
    lx = Lx(a, n)
    f = F(a, n)
    return Matrix(n, n, lambda i, j: lx[i, 0].coeff(x, j) / f[i]).T


def Bt(a, n):
    return L(a, n) * T(a, n)


def B(a, n):
    return Bt(a, n - 1).row_insert(
        n - 1, Matrix(1, n, lambda i, j: 1 if j == n - 1 else 0)
    )


FractionsInG = 0
FractionsInA = 1
FractionsInB = 2
FractionsInF = 3


def cookToomFilter(
    a: Sequence,
    n: int,
    r: int,
    fractionsIn: int = FractionsInG,
    precision: int = None,
) -> Tuple[Matrix, Matrix, Matrix, Matrix]:
    """Compute the Cook-Toom filter transforms for F(n, r).

    Given interpolation points *a*, output size *n*, and filter size *r*,
    returns the transform matrices (AT, G, BT, f) that implement an
    F(n, r) minimal filtering or convolution algorithm.

    Parameters
    ----------
    a : sequence
        Interpolation points.  Must contain at least ``n + r - 2`` elements.
    n : int
        Output size.
    r : int
        Filter (kernel) size.
    fractionsIn : int, optional
        Controls where rational fractions appear in the transforms.
        One of ``FractionsInG`` (default), ``FractionsInA``,
        ``FractionsInB``, or ``FractionsInF``.
    precision : int, optional
        If given, compute the transforms symbolically first for numerical
        accuracy, then convert all matrix entries to floating-point values
        with the specified number of significant decimal digits.  Use 3
        for half precision (float16), 7 for single precision (float32),
        or 15 for double precision (float64).  Default is ``None``
        (return exact symbolic matrices).

    Returns
    -------
    (AT, G, BT, f) : tuple of sympy.Matrix
    """
    alpha = n + r - 1
    f = FdiagPlus1(a, alpha)
    if f[0, 0] < 0:
        f[0, :] *= -1
    if fractionsIn == FractionsInG:
        AT = A(a, alpha, n).T
        G = (A(a, alpha, r).T * f ** (-1)).T
        BT = f * B(a, alpha).T
    elif fractionsIn == FractionsInA:
        BT = f * B(a, alpha).T
        G = A(a, alpha, r)
        AT = (A(a, alpha, n)).T * f ** (-1)
    elif fractionsIn == FractionsInB:
        AT = A(a, alpha, n).T
        G = A(a, alpha, r)
        BT = B(a, alpha).T
    else:
        AT = A(a, alpha, n).T
        G = A(a, alpha, r)
        BT = f * B(a, alpha).T
    if precision is not None:
        AT = N(simplify(AT), precision)
        G = N(simplify(G), precision)
        BT = N(simplify(BT), precision)
        f = N(simplify(f), precision)
    return (AT, G, BT, f)


def filterVerify(n: int, r: int, AT: Matrix, G: Matrix, BT: Matrix) -> Matrix:
    """Symbolically verify an FIR filter by computing AT * ((G*g) . (BT*d)).

    Returns a column vector of symbolic expressions that should equal the
    linear convolution of *d* and *g*.
    """
    alpha = n + r - 1

    di = IndexedBase("d")
    gi = IndexedBase("g")
    d = Matrix(alpha, 1, lambda i, j: di[i])
    g = Matrix(r, 1, lambda i, j: gi[i])

    V = BT * d
    U = G * g
    M = U.multiply_elementwise(V)
    Y = simplify(AT * M)

    return Y


def convolutionVerify(n: int, r: int, B: Matrix, G: Matrix, A: Matrix) -> Matrix:
    """Symbolically verify a linear convolution by computing B * ((G*g) . (A*d)).

    Returns a column vector of symbolic expressions that should equal the
    linear convolution of *d* and *g*.
    """
    di = IndexedBase("d")
    gi = IndexedBase("g")

    d = Matrix(n, 1, lambda i, j: di[i])
    g = Matrix(r, 1, lambda i, j: gi[i])

    V = A * d
    U = G * g
    M = U.multiply_elementwise(V)
    Y = simplify(B * M)

    return Y


def showCookToomFilter(
    a: Sequence, n: int, r: int, fractionsIn: int = FractionsInG
) -> None:
    """Print the Cook-Toom FIR filter transforms for F(n, r).

    Displays the AT, G, and BT matrices and, unless *fractionsIn* is
    ``FractionsInF``, symbolically verifies the result.
    """
    AT, G, BT, f = cookToomFilter(a, n, r, fractionsIn)

    print("AT = ")
    pprint(AT)
    print("")

    print("G = ")
    pprint(G)
    print("")

    print("BT = ")
    pprint(BT)
    print("")

    if fractionsIn != FractionsInF:
        print("FIR filter: AT*((G*g)(BT*d)) =")
        pprint(filterVerify(n, r, AT, G, BT))
        print("")

    if fractionsIn == FractionsInF:
        print("fractions = ")
        pprint(f)
        print("")


def showCookToomConvolution(
    a: Sequence, n: int, r: int, fractionsIn: int = FractionsInG
) -> None:
    """Print the Cook-Toom linear convolution transforms for F(n, r).

    Displays the A, G, and B matrices (transposed from the filter form)
    and, unless *fractionsIn* is ``FractionsInF``, symbolically verifies
    the result.
    """
    AT, G, BT, f = cookToomFilter(a, n, r, fractionsIn)

    B = BT.transpose()
    A = AT.transpose()

    print("A = ")
    pprint(A)
    print("")

    print("G = ")
    pprint(G)
    print("")

    print("B = ")
    pprint(B)
    print("")

    if fractionsIn != FractionsInF:
        print("Linear Convolution: B*((G*g)(A*d)) =")
        pprint(convolutionVerify(n, r, B, G, A))
        print("")

    if fractionsIn == FractionsInF:
        print("fractions = ")
        pprint(f)
        print("")


def chebyshevPoints(n: int, precision: int = None) -> tuple:
    """Return *n* Chebyshev nodes of the first kind.

    The Chebyshev nodes are the roots of the degree-*n* Chebyshev polynomial
    of the first kind, given by ``cos((2*k + 1) * pi / (2*n))`` for
    ``k = 0, 1, ..., n - 1``.

    These points minimise the maximum interpolation error (Runge phenomenon)
    and can be passed directly as the interpolation-point sequence *a* to
    :func:`cookToomFilter` or :func:`showCookToomFilter`.

    Parameters
    ----------
    n : int
        Number of interpolation points to generate.  For an F(m, r)
        algorithm you need ``n = m + r - 2`` points.
    precision : int, optional
        If given, return floating-point values with the specified number
        of significant decimal digits.  Use 3 for half precision
        (float16), 7 for single precision (float32), or 15 for double
        precision (float64).  Default is ``None`` (return exact symbolic
        expressions).

    Returns
    -------
    tuple
        A tuple of *n* values.  Each value is an exact SymPy ``cos(...)``
        expression when *precision* is ``None``, or a SymPy ``Float``
        when *precision* is specified.  Nodes are ordered from ``k = 0``
        (closest to 1) to ``k = n - 1`` (closest to -1).
    """
    if precision is not None:
        return tuple(
            Float(math.cos((2 * k + 1) * math.pi / (2 * n)), precision)
            for k in range(n)
        )
    return tuple(cos(Rational(2 * k + 1, 2 * n) * pi) for k in range(n))
