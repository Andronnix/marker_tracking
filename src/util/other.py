import cv2
import numpy as np
from typing import Iterator
from itertools import chain, zip_longest


def mult(m1, m2):
    """Multiplies two affine transformation matrices of 2x3 size"""
    assert np.shape(m1) == (2, 3)
    assert np.shape(m2) == (2, 3)

    m1_temp = np.vstack((m1, [0, 0, 1]))
    m2_temp = np.vstack((m2, [0, 0, 1]))
    result = m1_temp * m2_temp

    return result[:2, :]


def flip_points(a):
    """[(x, y), ...] -> [(y, x), ...]"""
    a = np.array(a)
    return np.flip(a, 1)


def transform32(points, H, add=(0, 0)):
    """Transform list of points [(x, y), ...]
        :return H * points.T() + add
    """
    points = np.float32(points)
    return np.int32(cv2.perspectiveTransform(points.reshape(1, -1, 2), H).reshape(-1, 2) + add)


def flatmap(func, *iterable) -> Iterator:
    """Flattens iterable and applies func to each element
       [[a, b], [c, d]] -> f(a), f(b), f(c), f(d)
    """
    return map(func, chain(*chain(*iterable)))


def flatmap2(func, *iterable) -> Iterator:
    """Flattens iterable and applies func to each element
       [[[a, b]], [[c, d]]] -> f(a), f(b), f(c), f(d)
    """
    return map(func, chain(*chain(*chain(*iterable))))


def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks
    grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx

    """
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)
