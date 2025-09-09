# -*- coding: utf-8 -*-
import re
from typing import Dict, Any

def parse_instruction(instr: str) -> Dict[str, Any]:
    out = {'margin_m': None, 'side_stop_m': None, 'mentions': []}
    m = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*(厘米|cm|米|m)', instr)
    if m:
        val = float(m.group(1)); unit = m.group(2)
        out['margin_m'] = val/100.0 if unit in ('厘米','cm') else val
    m2 = re.search(r'右侧\s*([0-9]+(?:\.[0-9]+)?)\s*(米|m)\s*停', instr)
    if m2: out['side_stop_m'] = float(m2.group(1))
    for k in [('人群','people'),('行人','people'),('桌','table'),('绕开','bypass'),('绕过','bypass')]:
        if k[0] in instr: out['mentions'].append(k[1])
    return out
