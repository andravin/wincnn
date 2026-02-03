import operator
from functools import reduce
from typing import Sequence, Tuple

from sympy import (
    IndexedBase,
    Matrix,
    Poly,
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
    a: Sequence, n: int, r: int, fractionsIn: int = FractionsInG
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
