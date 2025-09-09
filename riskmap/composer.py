# -*- coding: utf-8 -*-
import numpy as np
def fuse_risk_and_inflation(risk: np.ndarray, inflation: np.ndarray) -> np.ndarray:
    return np.clip(risk + inflation*0.5, 0.0, 1.5)
