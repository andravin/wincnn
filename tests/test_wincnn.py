from sympy import Matrix, Rational, IndexedBase, simplify

import wincnn


def test_f2x3_filter():
    """F(2,3) with interpolation points (0, 1, -1)."""
    AT, G, BT, f = wincnn.cookToomFilter((0, 1, -1), 2, 3)

    expected_AT = Matrix([
        [1, 1,  1, 0],
        [0, 1, -1, 1],
    ])
    expected_G = Matrix([
        [1,            0,            0],
        [Rational(1,2), Rational(1,2), Rational(1,2)],
        [Rational(1,2), Rational(-1,2), Rational(1,2)],
        [0,            0,            1],
    ])
    expected_BT = Matrix([
        [1,  0, -1, 0],
        [0,  1,  1, 0],
        [0, -1,  1, 0],
        [0, -1,  0, 1],
    ])

    assert AT == expected_AT
    assert G == expected_G
    assert BT == expected_BT


def test_f4x3_filter():
    """F(4,3) with interpolation points (0, 1, -1, 2, -2)."""
    AT, G, BT, f = wincnn.cookToomFilter((0, 1, -1, 2, -2), 4, 3)

    expected_AT = Matrix([
        [1, 1,  1, 1,  1, 0],
        [0, 1, -1, 2, -2, 0],
        [0, 1,  1, 4,  4, 0],
        [0, 1, -1, 8, -8, 1],
    ])
    expected_G = Matrix([
        [Rational(1,4),   0,            0],
        [Rational(-1,6),  Rational(-1,6), Rational(-1,6)],
        [Rational(-1,6),  Rational(1,6),  Rational(-1,6)],
        [Rational(1,24),  Rational(1,12), Rational(1,6)],
        [Rational(1,24),  Rational(-1,12), Rational(1,6)],
        [0,              0,              1],
    ])
    expected_BT = Matrix([
        [4,  0, -5,  0, 1, 0],
        [0, -4, -4,  1, 1, 0],
        [0,  4, -4, -1, 1, 0],
        [0, -2, -1,  2, 1, 0],
        [0,  2, -1, -2, 1, 0],
        [0,  4,  0, -5, 0, 1],
    ])

    assert AT == expected_AT
    assert G == expected_G
    assert BT == expected_BT


def test_f6x3_filter():
    """F(6,3) with interpolation points (0, 1, -1, 2, -2, 1/2, -1/2)."""
    AT, G, BT, f = wincnn.cookToomFilter(
        (0, 1, -1, 2, -2, Rational(1, 2), -Rational(1, 2)), 6, 3
    )

    expected_AT = Matrix([
        [1, 1,  1,  1,   1,   1,          1,         0],
        [0, 1, -1,  2,  -2,   Rational(1,2),  Rational(-1,2),  0],
        [0, 1,  1,  4,   4,   Rational(1,4),  Rational(1,4),   0],
        [0, 1, -1,  8,  -8,   Rational(1,8),  Rational(-1,8),  0],
        [0, 1,  1, 16,  16,   Rational(1,16), Rational(1,16),  0],
        [0, 1, -1, 32, -32,   Rational(1,32), Rational(-1,32), 1],
    ])
    expected_G = Matrix([
        [1,              0,               0],
        [Rational(-2,9), Rational(-2,9),  Rational(-2,9)],
        [Rational(-2,9), Rational(2,9),   Rational(-2,9)],
        [Rational(1,90), Rational(1,45),  Rational(2,45)],
        [Rational(1,90), Rational(-1,45), Rational(2,45)],
        [Rational(32,45), Rational(16,45), Rational(8,45)],
        [Rational(32,45), Rational(-16,45), Rational(8,45)],
        [0,              0,               1],
    ])
    expected_BT = Matrix([
        [1,          0, Rational(-21,4), 0,          Rational(21,4),  0,           -1, 0],
        [0,          1, 1,               Rational(-17,4), Rational(-17,4), 1,      1,  0],
        [0,         -1, 1,               Rational(17,4),  Rational(-17,4), -1,     1,  0],
        [0, Rational(1,2), Rational(1,4), Rational(-5,2), Rational(-5,4), 2,      1,  0],
        [0, Rational(-1,2), Rational(1,4), Rational(5,2), Rational(-5,4), -2,     1,  0],
        [0,          2, 4,               Rational(-5,2),  -5,              Rational(1,2), 1, 0],
        [0,         -2, 4,               Rational(5,2),   -5,              Rational(-1,2), 1, 0],
        [0,         -1, 0,               Rational(21,4),  0,              Rational(-21,4), 0, 1],
    ])

    assert AT == expected_AT
    assert G == expected_G
    assert BT == expected_BT


def test_f2x3_convolution():
    """Linear convolution F(2,3) — verify transposed A, G, B."""
    AT, G, BT, f = wincnn.cookToomFilter((0, 1, -1), 2, 3)

    A = AT.T
    B = BT.T

    expected_A = Matrix([
        [1,  0],
        [1,  1],
        [1, -1],
        [0,  1],
    ])
    expected_B = Matrix([
        [1,  0,  0,  0],
        [0,  1, -1, -1],
        [-1, 1,  1,  0],
        [0,  0,  0,  1],
    ])

    assert A == expected_A
    assert G == Matrix([
        [1,            0,            0],
        [Rational(1,2), Rational(1,2), Rational(1,2)],
        [Rational(1,2), Rational(-1,2), Rational(1,2)],
        [0,            0,            1],
    ])
    assert B == expected_B


def test_filter_verify():
    """Verify the symbolic FIR filter output for F(2,3)."""
    AT, G, BT, f = wincnn.cookToomFilter((0, 1, -1), 2, 3)
    Y = wincnn.filterVerify(2, 3, AT, G, BT)

    di = IndexedBase("d")
    gi = IndexedBase("g")

    expected = Matrix([
        [di[0]*gi[0] + di[1]*gi[1] + di[2]*gi[2]],
        [di[1]*gi[0] + di[2]*gi[1] + di[3]*gi[2]],
    ])

    assert simplify(Y - expected) == Matrix([[0], [0]])


def test_convolution_verify():
    """Verify the symbolic linear convolution output for F(2,3)."""
    AT, G, BT, f = wincnn.cookToomFilter((0, 1, -1), 2, 3)
    B = BT.T
    A = AT.T
    Y = wincnn.convolutionVerify(2, 3, B, G, A)

    di = IndexedBase("d")
    gi = IndexedBase("g")

    expected = Matrix([
        [di[0]*gi[0]],
        [di[0]*gi[1] + di[1]*gi[0]],
        [di[0]*gi[2] + di[1]*gi[1]],
        [di[1]*gi[2]],
    ])

    assert simplify(Y - expected) == Matrix([[0], [0], [0], [0]])
