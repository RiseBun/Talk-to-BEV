# -*- coding: utf-8 -*-
import numpy as np
def pseudo_variance_from_risk(risk: np.ndarray) -> np.ndarray:
    return np.clip(risk * 0.3, 0.0, 0.5)
