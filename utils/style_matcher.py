import math
from typing import Dict, Iterable, List, Optional, Set, Tuple

import cv2
import numpy as np

from . import shared
from .fontformat import FontFormat, TextAlignment, px2pt
from .textblock import TextBlock


MIN_COLOR_CONFIDENCE = 0.15
MIN_STROKE_CONFIDENCE = 0.25


def _valid_rgb(value) -> bool:
    if value is None:
        return False
    try:
        return len(value) >= 3 and all(float(v) >= 0 for v in value[:3])
    except Exception:
        return False


def _rgb_list(value) -> List[int]:
    return [int(np.clip(round(float(v)), 0, 255)) for v in value[:3]]


def _ensure_rgb_image(img: np.ndarray) -> Optional[np.ndarray]:
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.ndim == 3 and img.shape[-1] == 4:
        return np.ascontiguousarray(img[..., :3])
    if img.ndim == 3 and img.shape[-1] >= 3:
        return np.ascontiguousarray(img[..., :3])
    return None


def _font_size_from_geometry(blk: TextBlock) -> float:
    for value in (getattr(blk, '_detected_font_size', -1), getattr(blk, 'font_size', -1)):
        try:
            value = float(value)
            if value > 0:
                return value
        except Exception:
            pass

    try:
        lines = blk.lines_array(dtype=np.float64)
        if len(lines) > 0:
            middle = (lines[:, [1, 2, 3, 0]] + lines) / 2
            vec_v = middle[:, 2] - middle[:, 0]
            vec_h = middle[:, 1] - middle[:, 3]
            if blk.src_is_vertical:
                return max(float(np.linalg.norm(vec_h.sum(axis=0)) / max(len(lines), 1)), 1.0)
            return max(float(np.linalg.norm(vec_v.sum(axis=0)) / max(len(lines), 1)), 1.0)
    except Exception:
        pass
    return 24.0


def _line_polygons(blk: TextBlock) -> np.ndarray:
    try:
        lines = blk.lines_array(dtype=np.float64)
        if lines.size > 0:
            return lines.reshape(-1, 4, 2)
    except Exception:
        pass

    try:
        x, y, w, h = blk.bounding_rect()
        return np.array([[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]], dtype=np.float64)
    except Exception:
        return np.zeros((0, 4, 2), dtype=np.float64)


def _crop_and_line_mask(img: np.ndarray, blk: TextBlock, font_size: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    polygons = _line_polygons(blk)
    if polygons.size == 0:
        return None, None

    h, w = img.shape[:2]
    pad = max(int(round(font_size * 0.35)), 4)
    x1 = max(int(math.floor(polygons[..., 0].min())) - pad, 0)
    y1 = max(int(math.floor(polygons[..., 1].min())) - pad, 0)
    x2 = min(int(math.ceil(polygons[..., 0].max())) + pad, w)
    y2 = min(int(math.ceil(polygons[..., 1].max())) + pad, h)
    if x2 <= x1 or y2 <= y1:
        return None, None

    crop = img[y1:y2, x1:x2]
    shifted = np.round(polygons - np.array([x1, y1], dtype=np.float64)).astype(np.int32)
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, shifted, 255)

    if cv2.countNonZero(mask) == 0:
        mask[:, :] = 255
    return crop, mask


def _border_mask(shape: Tuple[int, int], width: int = 3) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    bw = min(max(width, 1), max(min(h, w) // 2, 1))
    mask[:bw, :] = True
    mask[-bw:, :] = True
    mask[:, :bw] = True
    mask[:, -bw:] = True
    return mask


def _median_rgb(img: np.ndarray, mask: np.ndarray) -> Optional[List[int]]:
    if mask is None or np.count_nonzero(mask) == 0:
        return None
    pixels = img[mask]
    if pixels.size == 0:
        return None
    return _rgb_list(np.median(pixels[:, :3], axis=0))


def _lab_color(rgb: Iterable[int]) -> np.ndarray:
    arr = np.array([[list(rgb)[:3]]], dtype=np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)


def _lab_delta(rgb1: Iterable[int], rgb2: Iterable[int]) -> float:
    return float(np.linalg.norm(_lab_color(rgb1) - _lab_color(rgb2)))


def _lab_distance_map(lab_img: np.ndarray, rgb: Iterable[int]) -> np.ndarray:
    return np.linalg.norm(lab_img.astype(np.float32) - _lab_color(rgb), axis=2)


def _clean_text_mask(mask: np.ndarray, line_mask: np.ndarray) -> np.ndarray:
    mask = np.logical_and(mask, line_mask > 0).astype(np.uint8) * 255
    if cv2.countNonZero(mask) == 0:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    min_area = max(2, int(cv2.countNonZero(line_mask) * 0.0005))
    cleaned = np.zeros_like(mask)
    for lab in range(1, num_labels):
        if stats[lab, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == lab] = 255
    return cleaned


def _extract_text_mask(crop: np.ndarray, line_mask: np.ndarray, bg_rgb: List[int]) -> np.ndarray:
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    dist = _lab_distance_map(lab, bg_rgb)
    line_bool = line_mask > 0
    vals = dist[line_bool]
    if vals.size == 0:
        return np.zeros(line_mask.shape, dtype=np.uint8)

    if vals.max() - vals.min() > 2:
        vals_u8 = np.clip(vals, 0, 255).astype(np.uint8)
        otsu_thr, _ = cv2.threshold(vals_u8.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = max(10.0, min(float(otsu_thr), 45.0))
        text_bool = np.logical_and(line_bool, dist > threshold)
    else:
        text_bool = np.zeros(line_mask.shape, dtype=bool)

    min_pixels = max(8, int(cv2.countNonZero(line_mask) * 0.01))
    if np.count_nonzero(text_bool) < min_pixels:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        bg_lum = 0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]
        line_gray = gray[line_bool]
        if line_gray.size > 0:
            if bg_lum >= 127:
                threshold = np.percentile(line_gray, 35)
                text_bool = np.logical_and(line_bool, gray <= threshold)
            else:
                threshold = np.percentile(line_gray, 65)
                text_bool = np.logical_and(line_bool, gray >= threshold)

    return _clean_text_mask(text_bool, line_mask)


def _estimate_alignment(blk: TextBlock) -> int:
    try:
        if blk.src_is_vertical:
            return int(TextAlignment.Center)
        angled, center, polygons = blk.unrotated_polygons()
        polygons = polygons.reshape(-1, 4, 2)
        if len(polygons) <= 1:
            return int(blk.alignment)
        left_std = np.std(polygons[:, 0, 0])
        right_std = np.std(polygons[:, 1, 0])
        center_std = np.std((polygons[:, 0, 0] + polygons[:, 1, 0]) / 2) * 0.7
        if left_std < right_std and left_std < center_std:
            return int(TextAlignment.Left)
        if right_std < left_std and right_std < center_std:
            return int(TextAlignment.Right)
        return int(TextAlignment.Center)
    except Exception:
        return int(getattr(blk, 'alignment', TextAlignment.Left))


def _source_char_width_ratio(blk: TextBlock, font_size: float) -> Optional[float]:
    if font_size <= 0:
        return None
    try:
        text = blk.get_text()
        char_count = len([c for c in text if not c.isspace()])
        if char_count <= 0:
            return None
        _, width = blk.normalizd_width_list(normalize=False)
        return float(width) / char_count / font_size
    except Exception:
        return None


def extract_source_style(img: np.ndarray, blk: TextBlock) -> Dict:
    """Estimate source text style from the original page image.

    The result is intentionally best-effort. It is stored on TextBlock and later
    applied only for options still configured as "decide by program".
    """
    font_size = _font_size_from_geometry(blk)
    style = {
        'font_size_px': float(font_size),
        'fill_rgb': _rgb_list(getattr(blk, 'fg_colors', [0, 0, 0])),
        'stroke_rgb': _rgb_list(getattr(blk, 'bg_colors', [0, 0, 0])),
        'stroke_width_ratio': 0.0,
        'alignment': _estimate_alignment(blk),
        'font_family_candidate': getattr(blk, '_detected_font_name', '') or '',
        'font_confidence': float(getattr(blk, '_detected_font_confidence', 0.0) or 0.0),
        'source_char_width_ratio': _source_char_width_ratio(blk, font_size),
        'confidence': 0.0,
    }

    rgb_img = _ensure_rgb_image(img)
    if rgb_img is None:
        return style

    crop, line_mask = _crop_and_line_mask(rgb_img, blk, font_size)
    if crop is None or line_mask is None:
        return style

    kernel = np.ones((3, 3), dtype=np.uint8)
    bg_mask = cv2.dilate(line_mask, kernel, iterations=1) == 0
    if np.count_nonzero(bg_mask) < 8:
        bg_mask = _border_mask(crop.shape[:2])
    bg_rgb = _median_rgb(crop, bg_mask)
    if bg_rgb is None:
        bg_rgb = _median_rgb(crop, line_mask > 0)
    if bg_rgb is None:
        return style

    text_mask = _extract_text_mask(crop, line_mask, bg_rgb)
    text_area = cv2.countNonZero(text_mask)
    line_area = max(cv2.countNonZero(line_mask), 1)
    if text_area < max(8, int(line_area * 0.005)):
        style['confidence'] = 0.05
        return style

    erode_iters = max(1, int(round(font_size / 48.0)))
    inner_mask = cv2.erode(text_mask, kernel, iterations=erode_iters) > 0
    if np.count_nonzero(inner_mask) < max(8, int(text_area * 0.15)):
        inner_mask = text_mask > 0

    fill_rgb = _median_rgb(crop, inner_mask)
    if fill_rgb is not None:
        style['fill_rgb'] = fill_rgb

    one_px_inner = cv2.erode(text_mask, kernel, iterations=1) > 0
    boundary_mask = np.logical_and(text_mask > 0, np.logical_not(one_px_inner))
    boundary_rgb = _median_rgb(crop, boundary_mask)
    if boundary_rgb is None:
        boundary_rgb = bg_rgb

    stroke_ratio = 0.0
    stroke_rgb = bg_rgb
    if fill_rgb is not None and boundary_rgb is not None:
        fill_boundary_delta = _lab_delta(fill_rgb, boundary_rgb)
        boundary_bg_delta = _lab_delta(boundary_rgb, bg_rgb)
        if fill_boundary_delta > 18 and boundary_bg_delta > 10:
            lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
            boundary_dist = _lab_distance_map(lab, boundary_rgb)
            fill_dist = _lab_distance_map(lab, fill_rgb)
            stroke_like = np.logical_and(text_mask > 0, boundary_dist < fill_dist)
            if np.count_nonzero(stroke_like) < max(4, int(text_area * 0.03)):
                stroke_like = boundary_mask
            dist_transform = cv2.distanceTransform(text_mask, cv2.DIST_L2, 3)
            stroke_width_px = float(np.percentile(dist_transform[stroke_like], 75)) if np.count_nonzero(stroke_like) else 0.0
            if stroke_width_px >= max(1.2, font_size * 0.035):
                stroke_ratio = min(stroke_width_px / max(font_size, 1.0), 0.35)
                stroke_rgb = boundary_rgb

    style['stroke_rgb'] = _rgb_list(stroke_rgb)
    style['stroke_width_ratio'] = float(stroke_ratio)

    contrast = min(np.median(_lab_distance_map(cv2.cvtColor(crop, cv2.COLOR_RGB2LAB), bg_rgb)[text_mask > 0]) / 80.0, 1.0)
    coverage = min(text_area / line_area * 2.5, 0.35)
    style['confidence'] = float(np.clip(0.2 + contrast * 0.45 + coverage, 0.0, 1.0))
    return style


def _normalized_name(name: str) -> str:
    return ''.join(ch for ch in str(name).lower() if ch.isalnum())


def _dedupe(items: Iterable[str]) -> List[str]:
    result = []
    seen = set()
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _available_fonts() -> List[str]:
    fonts = []
    custom_fonts = getattr(shared, 'CUSTOM_FONTS', None) or []
    fonts.extend(custom_fonts)
    all_fonts = getattr(shared, 'FONT_FAMILIES', None) or []
    fonts.extend(sorted(all_fonts))
    return _dedupe(fonts)


def _match_installed_font(name: str) -> Optional[str]:
    if not name:
        return None
    fonts = _available_fonts()
    if not fonts:
        return None
    target = _normalized_name(name)
    if not target or target == 'unknown':
        return None

    for family in fonts:
        if family.lower() == str(name).lower():
            return family
    for family in fonts:
        normalized = _normalized_name(family)
        if normalized == target or target in normalized or normalized in target:
            return family
    return None


def _font_metric_candidate(blk: TextBlock, style: Dict, current_family: str) -> Optional[str]:
    custom_fonts = getattr(shared, 'CUSTOM_FONTS', None) or []
    all_fonts = getattr(shared, 'FONT_FAMILIES', None) or []
    candidates = _dedupe(custom_fonts if len(custom_fonts) > 0 else sorted(all_fonts))
    if not candidates:
        return None

    try:
        from qtpy.QtGui import QFont, QFontMetricsF
        from qtpy.QtWidgets import QApplication
    except Exception:
        return None

    if QApplication.instance() is None:
        return None

    sample = blk.get_text()
    sample = ''.join(ch for ch in sample if not ch.isspace())
    if not sample:
        sample = '漢字Aa'
    sample = sample[:24]

    font_size = float(style.get('font_size_px') or _font_size_from_geometry(blk))
    target_ratio = style.get('source_char_width_ratio')
    if target_ratio is None or target_ratio <= 0:
        return candidates[0] if candidates else None

    best_family = None
    best_score = float('inf')
    for family in candidates:
        try:
            font = QFont(family)
            font.setPointSizeF(px2pt(font_size))
            metrics = QFontMetricsF(font)
            width_ratio = float(metrics.horizontalAdvance(sample)) / max(len(sample), 1) / max(font_size, 1.0)
            height_ratio = float(metrics.height()) / max(font_size, 1.0)
            score = abs(width_ratio - target_ratio) + abs(height_ratio - 1.2) * 0.1
            if family == current_family:
                score += 0.03
            if score < best_score:
                best_score = score
                best_family = family
        except Exception:
            continue
    return best_family


def _should_replace_family(blk: TextBlock) -> bool:
    family = getattr(blk, 'font_family', '') or ''
    default_families = {shared.APP_DEFAULT_FONT, shared.DEFAULT_FONT_FAMILY, ''}
    return family in default_families


def _resolve_font_family(blk: TextBlock, style: Dict) -> Optional[str]:
    detected_conf = float(style.get('font_confidence') or getattr(blk, '_detected_font_confidence', 0.0) or 0.0)
    detected_name = style.get('font_family_candidate') or getattr(blk, '_detected_font_name', '')
    if detected_conf >= 0.6:
        matched = _match_installed_font(detected_name)
        if matched is not None:
            return matched
    return _font_metric_candidate(blk, style, getattr(blk, 'font_family', '') or '')


def apply_smart_style(blk: TextBlock, global_format: FontFormat, program_config) -> Set[str]:
    style = getattr(blk, 'source_style', None) or {}
    if not isinstance(style, dict):
        return set()

    updated: Set[str] = set()
    confidence = float(style.get('confidence') or 0.0)

    if getattr(program_config, 'let_fntsize_flag', 0) == 0:
        font_size = float(style.get('font_size_px') or 0.0)
        if font_size > 0:
            blk.font_size = font_size
            updated.add('font_size')

    if confidence >= MIN_COLOR_CONFIDENCE and getattr(program_config, 'let_fntcolor_flag', 0) == 0:
        fill_rgb = style.get('fill_rgb')
        if _valid_rgb(fill_rgb):
            blk.set_font_colors(fg_colors=_rgb_list(fill_rgb))
            updated.add('frgb')

    if confidence >= MIN_COLOR_CONFIDENCE and getattr(program_config, 'let_fnt_scolor_flag', 0) == 0:
        stroke_rgb = style.get('stroke_rgb')
        if _valid_rgb(stroke_rgb):
            blk.set_font_colors(bg_colors=_rgb_list(stroke_rgb))
            updated.add('srgb')

    if getattr(program_config, 'let_fntstroke_flag', 0) == 0:
        stroke_ratio = float(style.get('stroke_width_ratio') or 0.0)
        if confidence >= MIN_STROKE_CONFIDENCE and stroke_ratio >= 0.01:
            blk.stroke_width = stroke_ratio
        else:
            blk.stroke_width = 0.0
        updated.add('stroke_width')

    if getattr(program_config, 'let_alignment_flag', 0) == 0:
        alignment = style.get('alignment')
        if alignment in (int(TextAlignment.Left), int(TextAlignment.Center), int(TextAlignment.Right)):
            blk.alignment = int(alignment)
            updated.add('alignment')

    if getattr(program_config, 'let_family_flag', 0) == 0 and _should_replace_family(blk):
        family = _resolve_font_family(blk, style)
        if family:
            blk.font_family = family
            updated.add('font_family')

    return updated
