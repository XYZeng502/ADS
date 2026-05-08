from dataclasses import dataclass
from typing import Literal, Tuple

from app.config import settings
from app.schemas import ProductState


D1SignalSource = Literal["BLENDED", "ACTUAL", "PREDICTED_SLOT_WEIGHTED", "PREDICTED_SLOT_FALLBACK"]


@dataclass
class D1SignalMeta:
    raw_signal: float
    calibrated_signal: float
    lower_bound: float
    upper_bound: float


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _weighted_slot_pred_d1(product: ProductState) -> Tuple[float, D1SignalSource]:
    weighted_num = 0.0
    weighted_den = 0.0
    plain_vals: list[float] = []
    for slot in product.slots:
        v = max(slot.predicted_d1_roi, 0.0)
        plain_vals.append(v)
        w = max(slot.yesterday_spend, 0.0)
        if w > 0:
            weighted_num += w * v
            weighted_den += w
    if weighted_den > 0:
        return weighted_num / weighted_den, "PREDICTED_SLOT_WEIGHTED"
    return (_mean(plain_vals) if plain_vals else 0.0), "PREDICTED_SLOT_FALLBACK"


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    q = max(0.0, min(1.0, q))
    idx = int(round((len(v) - 1) * q))
    return v[idx]


def _calibrate_with_actual_history(raw_signal: float, actual: list[float]) -> D1SignalMeta:
    if not actual:
        clipped = max(settings.d1_signal_global_min, min(raw_signal, settings.d1_signal_global_max))
        return D1SignalMeta(raw_signal=raw_signal, calibrated_signal=clipped, lower_bound=settings.d1_signal_global_min, upper_bound=settings.d1_signal_global_max)

    actual_mean = _mean(actual)
    alpha = max(0.0, min(settings.d1_signal_shrink_to_actual_mean, 1.0))
    shrunk = (1.0 - alpha) * raw_signal + alpha * actual_mean

    ql = _quantile(actual, settings.d1_signal_clip_lower_quantile)
    qu = _quantile(actual, settings.d1_signal_clip_upper_quantile)
    if qu < ql:
        ql, qu = qu, ql
    span = max(qu - ql, 0.05)
    expand = max(settings.d1_signal_clip_expand_ratio, 0.0)
    lower = max(settings.d1_signal_global_min, ql - span * expand)
    upper = min(settings.d1_signal_global_max, qu + span * expand)
    if upper < lower:
        upper = lower
    calibrated = max(lower, min(shrunk, upper))
    return D1SignalMeta(raw_signal=raw_signal, calibrated_signal=calibrated, lower_bound=lower, upper_bound=upper)


def resolve_product_d1_signal_with_meta(product: ProductState) -> Tuple[float, D1SignalSource, D1SignalMeta]:
    min_points = max(settings.d1_signal_min_points, 1)
    actual = [float(x) for x in product.recent_d1_actual if x is not None and x > 0]
    pred = [float(x) for x in product.recent_d1_predicted if x is not None and x > 0]
    slot_pred, slot_source = _weighted_slot_pred_d1(product)

    raw_signal = 0.0
    source: D1SignalSource = slot_source
    if len(actual) >= min_points and len(pred) >= min_points:
        a = _mean(actual[-min_points:])
        p = _mean(pred[-min_points:])
        w_a = max(settings.d1_signal_actual_weight, 0.0)
        w_p = max(settings.d1_signal_pred_weight, 0.0)
        norm = w_a + w_p
        if norm <= 0:
            raw_signal, source = slot_pred, slot_source
        else:
            raw_signal, source = (w_a / norm) * a + (w_p / norm) * p, "BLENDED"
    elif len(actual) >= min_points:
        raw_signal, source = _mean(actual[-min_points:]), "ACTUAL"
    else:
        raw_signal, source = slot_pred, slot_source

    if not settings.d1_signal_calibration_enabled:
        meta = D1SignalMeta(
            raw_signal=raw_signal,
            calibrated_signal=raw_signal,
            lower_bound=settings.d1_signal_global_min,
            upper_bound=settings.d1_signal_global_max,
        )
        return raw_signal, source, meta

    meta = _calibrate_with_actual_history(raw_signal=raw_signal, actual=actual[-max(min_points, len(actual)) :])
    return meta.calibrated_signal, source, meta


def resolve_product_d1_signal(product: ProductState) -> Tuple[float, D1SignalSource]:
    calibrated, source, _ = resolve_product_d1_signal_with_meta(product)
    return calibrated, source
