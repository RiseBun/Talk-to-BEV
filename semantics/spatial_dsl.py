# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Tuple

def polygon_rect(cx: float, cy: float, w: float, h: float) -> List[Tuple[float, float]]:
    return [(cx - w/2, cy - h/2),
            (cx + w/2, cy - h/2),
            (cx + w/2, cy + h/2),
            (cx - w/2, cy + h/2)]

def build_patch_from_bev(lang_slots: Dict[str, Any],
                         bev_objects: List[Dict[str, Any]],
                         risk_map_path: str = None,
                         base_margin: float = 0.2) -> Dict[str, Any]:
    margin = lang_slots.get('margin_m') or base_margin
    masks: List[Dict[str, Any]] = []
    for obj in bev_objects:
        cls = obj.get('cls'); cx, cy, w, h = obj['cx'], obj['cy'], obj['w'], obj['h']
        if cls == 'people':
            masks.append({
                'id': f'ko_people_{len(masks)}',
                'shape': 'polygon', 'mode': 'keepout',
                'points': polygon_rect(cx, cy, w, h),
                'margin_m': margin, 'hardness': 'hybrid', 'weight': 10.0,
                'note': '绕开人群'
            })
        elif cls == 'table':
            masks.append({
                'id': f'ko_table_{len(masks)}',
                'shape': 'polygon', 'mode': 'keepout',
                'points': polygon_rect(cx, cy, w, h),
                'margin_m': margin, 'hardness': 'soft', 'weight': 5.0,
                'note': '靠近桌子更保守'
            })
    patch: Dict[str, Any] = {'spatial_masks': masks}
    if risk_map_path:
        patch['risk_map'] = {'type': 'grid','resolution_m': 0.1,'origin_xy': [0.0,0.0],'values': risk_map_path}
    return patch
