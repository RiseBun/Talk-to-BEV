# -*- coding: utf-8 -*-
import numpy as np
from typing import Tuple
def load_risk_map(path: str) -> Tuple[np.ndarray, float]:
    risk = np.load(path).astype('float32')
    return risk, 0.1
