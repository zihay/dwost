from diff_wost.core.fwd import *
import drjit as dr

# Constants for Bessel function implementations
ACC = 40.0
BIGNO = 1.0e10
BIGNI = 1.0e-10


@dr.syntax
def bessj0(x: Float) -> Float:
    """
    Evaluate Bessel function of first kind and order 0 at input x
    """
    ax = dr.abs(x)
    y = Float(0)
    ans = Float(0)
    ans1 = Float(0)
    ans2 = Float(0)

    mask = ax < 8.0

    if mask:
        y = x * x
        ans1 = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7
                                                          + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))))
        ans2 = 57568490411.0 + y * (1029532985.0 + y * (9494680.718
                                                        + y * (59272.64853 + y * (267.8532712 + y * 1.0))))
        ans = ans1 / ans2
    else:
        z = 8.0 / ax
        y = z * z
        xx = ax - 0.785398164
        ans1 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4
                                                  + y * (-0.2073370639e-5 + y * 0.2093887211e-6)))
        ans2 = -0.1562499995e-1 + y * (0.1430488765e-3
                                       + y * (-0.6911147651e-5 + y * (0.7621095161e-6
                                                                      - y * 0.934935152e-7)))
        ans = dr.sqrt(0.636619772 / ax) * (dr.cos(xx)
                                           * ans1 - z * dr.sin(xx) * ans2)

    return ans


@dr.syntax
def bessj1(x: Float) -> Float:
    """
    Evaluate Bessel function of first kind and order 1 at input x
    """
    ax = dr.abs(x)
    y = Float(0)
    ans = Float(0)
    ans1 = Float(0)
    ans2 = Float(0)

    if ax < 8.0:
        y = x * x
        ans1 = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1
                                                              + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))))
        ans2 = 144725228442.0 + y * (2300535178.0 + y * (18583304.74
                                                         + y * (99447.43394 + y * (376.9991397 + y * 1.0))))
        ans = ans1 / ans2
    else:
        z = 8.0 / ax
        y = z * z
        xx = ax - 2.356194491
        ans1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4
                                             + y * (0.2457520174e-5 + y * (-0.240337019e-6))))
        ans2 = 0.04687499995 + y * (-0.2002690873e-3
                                    + y * (0.8449199096e-5 + y * (-0.88228987e-6
                                                                  + y * 0.105787412e-6)))
        ans = dr.sqrt(0.636619772 / ax) * (dr.cos(xx)
                                           * ans1 - z * dr.sin(xx) * ans2)
        ans = dr.select(x < 0.0, -ans, ans)

    return ans


# @dr.syntax
# def bessj(n: int, x: Float) -> Float:
#     """
#     Evaluate Bessel function of first kind and integer order n at input x
#     """
#     ax = dr.abs(x)

#     # Special cases for n=0 and n=1
#     if n == 0:
#         return bessj0(ax)
#     if n == 1:
#         return bessj1(ax)

#     ret = Float(0.0)
#     # For x=0, return 0 for n>0
#     if ax == 0.0:
#         ret = Float(0.0)
#     elif ax > n:
#         tox = Float(2.0) / ax
#         bjm = bessj0(ax)
#         bj = bessj1(ax)
#         j = Int(1)
#         while j < n:
#             bjp = j * tox * bj - bjm
#             bjm = bj
#             bj = bjp
#             j += 1
#         ret = bj
#     else:
#         tox = Float(2.0) / ax
#         m = 2 * ((n + Int(dr.sqrt(ACC * n))) // 2)
#         jsum = Bool(False)
#         bjp = Float(0.0)
#         bj = Float(1.0)
#         sum = Float(0.0)

#         j = m
#         while j > 0:
#             bjm = j * tox * bj - bjp
#             bjp = bj
#             bj = bjm
#             if dr.abs(bj) > BIGNO:
#                 bj *= BIGNI
#                 bjp *= BIGNI
#                 ans *= BIGNI
#                 sum *= BIGNI

#             if jsum:
#                 sum += bj
#             jsum = ~jsum
#             if j == n:
#                 ans = bjp
#             j -= 1

#         sum = 2.0 * sum - bj
#         ans = ans / sum

#         ret = dr.select((x < 0.0) & (n % 2 == 1), -ans, ans)

#     return ret


@dr.syntax
def bessy0(x: Float) -> Float:
    """
    Evaluate Bessel function of second kind and order 0 at input x
    """
    ans = Float(0)

    if x < 8.0:
        y = x * x
        ans1 = -2957821389.0 + y * (7062834065.0 + y * (-512359803.6
                                                        + y * (10879881.29 + y * (-86327.92757 + y * 228.4622733))))
        ans2 = 40076544269.0 + y * (745249964.8 + y * (7189466.438
                                                       + y * (47447.26470 + y * (226.1030244 + y * 1.0))))
        ans = (ans1 / ans2) + 0.636619772 * bessj0(x) * dr.log(x)
    else:
        z = 8.0 / x
        y = z * z
        xx = x - 0.785398164
        ans1 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4
                                                  + y * (-0.2073370639e-5 + y * 0.2093887211e-6)))
        ans2 = -0.1562499995e-1 + y * (0.1430488765e-3
                                       + y * (-0.6911147651e-5 + y * (0.7621095161e-6
                                                                      + y * (-0.934945152e-7))))
        ans = dr.sqrt(0.636619772 / x) * (dr.sin(xx)
                                          * ans1 + z * dr.cos(xx) * ans2)

    return ans


@dr.syntax
def bessy1(x: Float) -> Float:
    """
    Evaluate Bessel function of second kind and order 1 at input x
    """
    ans = Float(0)

    if x < 8.0:
        y = x * x
        ans1 = x * (-0.4900604943e13 + y * (0.1275274390e13
                                            + y * (-0.5153438139e11 + y * (0.7349264551e9
                                                                           + y * (-0.4237922726e7 + y * 0.8511937935e4)))))
        ans2 = 0.2499580570e14 + y * (0.4244419664e12
                                      + y * (0.3733650367e10 + y * (0.2245904002e8
                                                                    + y * (0.1020426050e6 + y * (0.3549632885e3 + y)))))
        ans = (ans1 / ans2) + 0.636619772 * (bessj1(x) * dr.log(x) - 1.0 / x)
    else:
        z = 8.0 / x
        y = z * z
        xx = x - 2.356194491
        ans1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4
                                             + y * (0.2457520174e-5 + y * (-0.240337019e-6))))
        ans2 = 0.04687499995 + y * (-0.2002690873e-3
                                    + y * (0.8449199096e-5 + y * (-0.88228987e-6
                                                                  + y * 0.105787412e-6)))
        ans = dr.sqrt(0.636619772 / x) * (dr.sin(xx)
                                          * ans1 + z * dr.cos(xx) * ans2)

    return ans


# @dr.syntax
# def bessy(n: int, x: Float) -> Float:
#     """
#     Evaluate Bessel function of second kind and order n at input x
#     Note: this function is not defined for x = 0
#     """
#     # Special cases for n=0 and n=1
#     if n == 0:
#         return bessy0(x)
#     if n == 1:
#         return bessy1(x)

#     # Use forward recurrence
#     tox = Float(2.0) / x
#     by = bessy1(x)
#     bym = bessy0(x)

#     j = Int(1)
#     while j < n:
#         byp = j * tox * by - bym
#         bym = by
#         by = byp
#         j += 1

#     return by


@dr.syntax
def bessi0(x: Float) -> Float:
    """
    Evaluate modified Bessel function of first kind and order 0 at input x
    """
    ax = dr.abs(x)
    ans = Float(0)
    y = Float(0)
    if ax < 3.75:
        y = (x / 3.75) ** 2
        ans = 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492
                                                           + y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2)))))
    else:
        y = 3.75 / ax
        ans = (dr.exp(ax) / dr.sqrt(ax)) * (0.39894228 + y * (0.1328592e-1
                                                              + y * (0.225319e-2 + y * (-0.157565e-2 + y * (0.916281e-2
                                                                                                            + y * (-0.2057706e-1 + y * (0.2635537e-1 + y * (-0.1647633e-1
                                                                                                                                                            + y * 0.392377e-2))))))))
    return ans


@dr.syntax
def bessi1(x: Float) -> Float:
    """
    Evaluate modified Bessel function of first kind and order 1 at input x
    """
    ax = dr.abs(x)
    ans = Float(0)
    y = Float(0)
    if ax < 3.75:
        y = (x / 3.75) ** 2
        ans = ax * (0.5 + y * (0.87890594 + y * (0.51498869 + y * (0.15084934
                                                                   + y * (0.2658733e-1 + y * (0.301532e-2 + y * 0.32411e-3))))))
    else:
        y = 3.75 / ax
        ans = 0.2282967e-1 + y * (-0.2895312e-1 + y * (0.1787654e-1
                                                       - y * 0.420059e-2))
        ans = 0.39894228 + y * (-0.3988024e-1 + y * (-0.362018e-2
                                                     + y * (0.163801e-2 + y * (-0.1031555e-1 + y * ans))))
        ans *= (dr.exp(ax) / dr.sqrt(ax))

    # Apply sign adjustment for negative x
    return dr.select(x < 0.0, -ans, ans)


# @dr.syntax
# def bessi(n: int, x: Float) -> Float:
#     """
#     Evaluate modified Bessel function of first kind and order n at input x
#     """
#     # Special cases for n=0 and n=1
#     if n == 0:
#         return bessi0(x)
#     if n == 1:
#         return bessi1(x)

#     ret = Float(0.0)
#     # For x=0, return 0 for n>0
#     if x == 0.0:
#         ret = Float(0.0)
#     else:
#         tox = 2.0 / dr.abs(x)
#         bip = Float(0.0)
#         ans = Float(0.0)
#         bi = Float(1.0)

#         # Compute In using Miller's algorithm from higher order
#         j = 2 * (n + Int(dr.sqrt(ACC * n)))

#         while j > 0:
#             bim = bip + j * tox * bi
#             bip = bi
#             bi = bim

#             if dr.abs(bi) > BIGNO:
#                 ans *= BIGNI
#                 bi *= BIGNI
#                 bip *= BIGNI

#             if j == n:
#                 ans = bip

#             j -= 1

#         # Normalize with I0
#         ans *= bessi0(x) / bi

#         # Apply sign adjustment for negative x and odd order
#         ret = dr.select((x < 0.0) & (n % 2 == 1), -ans, ans)

#     return ret


@dr.syntax
def bessk0(x: Float) -> Float:
    """
    Evaluate modified Bessel function of second kind and order 0 at input x
    """
    ans = Float(0)
    y = Float(0)
    if x <= 2.0:
        y = x * x / 4.0
        ans = (-dr.log(x / 2.0) * bessi0(x)) + (-0.57721566 + y * (0.42278420
                                                                   + y * (0.23069756 + y * (0.3488590e-1 + y * (0.262698e-2
                                                                                                                + y * (0.10750e-3 + y * 0.74e-5))))))
    else:
        y = 2.0 / x
        ans = (dr.exp(-x) / dr.sqrt(x)) * (1.25331414 + y * (-0.7832358e-1
                                                             + y * (0.2189568e-1 + y * (-0.1062446e-1 + y * (0.587872e-2
                                                                                                             + y * (-0.251540e-2 + y * 0.53208e-3))))))

    return ans


@dr.syntax
def bessk1(x: Float) -> Float:
    """
    Evaluate modified Bessel function of second kind and order 1 at input x
    """
    ans = Float(0)
    y = Float(0)
    if x <= 2.0:
        y = x * x / 4.0
        ans = (dr.log(x / 2.0) * bessi1(x)) + (1.0 / x) * (1.0 + y * (0.15443144
                                                                      + y * (-0.67278579 + y * (-0.18156897 + y * (-0.1919402e-1
                                                                                                                   + y * (-0.110404e-2 + y * (-0.4686e-4)))))))
    else:
        y = 2.0 / x
        ans = (dr.exp(-x) / dr.sqrt(x)) * (1.25331414 + y * (0.23498619
                                                             + y * (-0.3655620e-1 + y * (0.1504268e-1 + y * (-0.780353e-2
                                                                                                             + y * (0.325614e-2 + y * (-0.68245e-3)))))))

    return ans


# @dr.syntax
# def bessk(n: int, x: Float) -> Float:
#     """
#     Evaluate modified Bessel function of second kind and order n at input x
#     Note: this function is not defined for x = 0
#     """
#     # Special cases for n=0 and n=1
#     if n == 0:
#         return bessk0(x)
#     if n == 1:
#         return bessk1(x)

#     # Use forward recurrence
#     tox = 2.0 / x
#     bkm = bessk0(x)
#     bk = bessk1(x)

#     j = Int(1)
#     while j < n:
#         bkp = bkm + j * tox * bk
#         bkm = bk
#         bk = bkp
#         j += 1

#     return bk
