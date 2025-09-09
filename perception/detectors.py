# -*- coding: utf-8 -*-
from typing import List, Dict, Any
def fake_bev_objects() -> List[Dict[str, Any]]:
    return [
        {'cls': 'people', 'cx': 2.4, 'cy': 3.0, 'w': 1.2, 'h': 1.0},
        {'cls': 'table',  'cx': 4.0, 'cy': 3.0, 'w': 1.0, 'h': 0.8},
    ]
