
import numpy as np


def f1(a, b ):
    return a + b


for i in range(5):
    x = f1(i, i * i)
    y = x
    print(y)
