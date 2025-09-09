# -*- coding: utf-8 -*-
import numpy as np
def compute_inflation(var: np.ndarray, risk: np.ndarray, speed: float,
                      a: float=0.6, b: float=0.2, c: float=0.4, cap: float=0.6) -> np.ndarray:
    inf = a*var + b*max(speed,0.0) + c*risk
    return np.clip(inf, 0.0, cap)
