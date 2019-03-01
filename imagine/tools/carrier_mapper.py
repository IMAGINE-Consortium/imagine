import numpy as np


def unity_mapper(_x, _a=0, _b=1):
    """
    Maps _x from [0, 1] into the interval [_a, _b]
    :param _x:
    :param _a:
    :param _b:
    :return:
    """
    return float(_x) * (float(_b)-float(_a)) + float(_a)
