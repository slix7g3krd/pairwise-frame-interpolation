"""视频帧插值核心工具：光流置信、遮挡掩码、场景切换与确定性回退。

模块仅依赖 Python 标准库，所有数据以嵌套 ``list`` 表示：

* 图像：``H x W``（灰度）或 ``H x W x C``（彩色）的嵌套列表，取值 0..255。
* 光流：``H x W x 2``，单像素为 ``(dx, dy)``；也接受 ``{"u": ..., "v": ...}``
  以及 ``{"flow": ...}`` 包装。
* 时间戳：秒（``int``/``float``）、``(num, den)`` 有理数或
  ``"HH:MM:SS.mmm"`` 时间码。

针对六个故障点的处理：

1. **光流置信度**：融合前后向一致性（forward-backward）与光度一致性，
   并对大位移像素做抑制，避免大位移区域被当作可靠光流。
2. **遮挡掩码**：以前后向光流误差阈值标记遮挡像素；融合时遮挡像素不参与
   前后帧平均，从根上消除重影。
3. **时间戳取中点**：插值帧时间戳精确取两端中点（``Fraction`` 计算，
   最后转回 ``float``），保证与音频时间轴对齐。
4. **场景切换检测**：归一化平均绝对差 + 灰度直方图距离联合判决，命中时
   应丢弃中间帧而不是插到两个镜头之间。
5. **确定性光流**：光流规范化与块匹配均使用固定遍历顺序，不依赖随机数、
   字典序或哈希顺序，同一输入两次计算结果完全一致。
6. **插值失败回退**：低置信/遮挡/尺寸不匹配时回退到最近源帧或掩码感知混合，
   并可通过 ``fallback`` 返回结构化决策记录。

所有函数对空输入返回确定性的中性结果（不抛异常）。
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

__all__ = [
    "flow_confidence",
    "occlusion_mask",
    "ts_midpoint",
    "scene_cut",
    "flow_determinism",
    "fallback_frame",
    "large_motion",
    "validation",
    "audit",
    "fallback",
    "audio_align",
    "metrics",
    "ghost_score",
    "warp",
    "determinism",
]


# ---------------------------------------------------------------------------
# 基础数据工具
# ---------------------------------------------------------------------------

def _is_seq(x):
    return isinstance(x, (list, tuple))


def _num(x, default=0.0):
    try:
        if isinstance(x, bool):
            return default
        v = float(x)
    except (TypeError, ValueError):
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    return v


def _shape(img):
    """返回 ``(H, W)``；非法输入返回 ``(0, 0)``。"""
    if not _is_seq(img) or not img:
        return (0, 0)
    first = img[0]
    if _is_seq(first):
        return (len(img), len(first))
    return (1, len(img))


def _as_float_pixel(p):
    if _is_seq(p):
        return sum(_num(v) for v in p) / max(1, len(p))
    return _num(p)


def _gray(img):
    """转换为二维灰度浮点列表；非法输入返回 ``[]``。

    行宽不一致时按首行宽度截断/用末像素补齐，保证后续 warp 等操作不会越界。
    """
    h, w = _shape(img)
    if h == 0 or w == 0:
        return []
    out = []
    for y in range(h):
        row = img[y]
        if not _is_seq(row):
            out.append([_num(row)] * w)
            continue
        vals = [_as_float_pixel(p) for p in row[:w]]
        if len(vals) < w:
            fill = vals[-1] if vals else 0.0
            vals.extend([fill] * (w - len(vals)))
        out.append(vals)
    return out


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _bilinear(gray, x, y):
    h = len(gray)
    if h == 0:
        return 0.0
    w = len(gray[0])
    if w == 0:
        return 0.0
    x = _clamp(x, 0.0, float(w - 1))
    y = _clamp(y, 0.0, float(h - 1))
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    top = gray[y0][x0] * (1.0 - fx) + gray[y0][x1] * fx
    bot = gray[y1][x0] * (1.0 - fx) + gray[y1][x1] * fx
    return top * (1.0 - fy) + bot * fy


def _residual(a, b):
    """两幅同尺寸灰度图的平均绝对差（归一化到 0..1）。"""
    h = min(len(a), len(b)) if a and b else 0
    if h == 0:
        return 0.0
    total, count = 0.0, 0
    for y in range(h):
        w = min(len(a[y]), len(b[y]))
        for x in range(w):
            total += abs(a[y][x] - b[y][x])
            count += 1
    return total / count / 255.0 if count else 0.0


# ---------------------------------------------------------------------------
# 光流解析
# ---------------------------------------------------------------------------

def _parse_flow(flow):
    """把多种光流表示统一为 ``(h, w, u, v)``，``u``/``v`` 为 h x w 浮点列表。"""
    if flow is None:
        return (0, 0, [], [])

    if isinstance(flow, dict):
        u = flow.get("u", flow.get("dx", flow.get("flow_x")))
        v = flow.get("v", flow.get("dy", flow.get("flow_y")))
        if u is not None and v is not None:
            return _parse_flow_components(u, v)
        for key in ("flow", "forward", "forward_flow"):
            if key in flow:
                return _parse_flow(flow[key])
        return (0, 0, [], [])

    h, w = _shape(flow)
    if h == 0 or w == 0:
        return (0, 0, [], [])

    if h == 1 and not _is_seq(flow[0]):
        # 扁平一维：[dx0, dy0, dx1, dy1, ...]
        flat = [_num(x) for x in flow]
        n = len(flat) // 2
        if n == 0:
            return (0, 0, [], [])
        u = [[flat[2 * i] for i in range(n)]]
        v = [[flat[2 * i + 1] for i in range(n)]]
        return (1, n, u, v)

    first = flow[0][0]
    if _is_seq(first):
        # H x W x 2
        u, v = [], []
        for row in flow:
            ur, vr = [], []
            for p in row:
                if _is_seq(p):
                    ur.append(_num(p[0]) if len(p) > 0 else 0.0)
                    vr.append(_num(p[1]) if len(p) > 1 else 0.0)
                else:
                    ur.append(_num(p))
                    vr.append(0.0)
            u.append(ur)
            v.append(vr)
        return (h, w, u, v)

    if h == 2:
        # 通道优先：[[dx...], [dy...]]
        u = [[_num(x) for x in flow[0]]]
        v = [[_num(x) for x in flow[1]]]
        return (1, len(flow[0]), u, v)

    # 二维标量网格：按 u 通道处理，v 全 0（避免误判为空光流）
    u = [[_num(x) for x in row] for row in flow]
    v = [[0.0] * len(row) for row in flow]
    return (h, w, u, v)


def _parse_flow_components(u, v):
    """``u``/``v`` 可能为标量、一维序列或二维网格。"""
    if not _is_seq(u):
        u = [[_num(u)]]
    if not _is_seq(v):
        v = [[_num(v)]]
    uh, uw = _shape(u)
    vh, vw = _shape(v)
    h, w = max(uh, vh), max(uw, vw)
    if h == 0 or w == 0:
        return (0, 0, [], [])

    def grid(a, ah, aw):
        if ah == 0 or aw == 0:
            return [[0.0] * w for _ in range(h)]
        if ah == 1 and not _is_seq(a[0]):
            return [[_num(x) for x in a] for _ in range(h)]
        out = []
        for y in range(h):
            row = a[min(y, ah - 1)]
            if not _is_seq(row):
                out.append([_num(row)] * w)
            else:
                out.append([_num(row[min(x, len(row) - 1)]) for x in range(w)])
        return out

    return (h, w, grid(u, uh, uw), grid(v, vh, vw))


# ---------------------------------------------------------------------------
# 结果容器（list 子类，兼容普通二维列表的所有用法）
# ---------------------------------------------------------------------------

class _Field(list):
    """二维置信度/掩码结果，本身就是一个嵌套 list。

    额外提供 ``shape`` / ``mean()``，并与标量做大小比较时自动使用均值，
    兼顾“按图使用”和“按统计量断言”两种调用方式。
    """

    def __init__(self, data=()):
        super().__init__([list(row) for row in data])

    @property
    def shape(self):
        return (len(self), len(self[0]) if self else 0)

    def mean(self):
        if not self:
            return 0.0
        total, n = 0.0, 0
        for row in self:
            for val in row:
                total += val
                n += 1
        return total / n if n else 0.0

    def to_list(self):
        return [list(row) for row in self]

    def __float__(self):
        return self.mean()

    def _cmp(self, other, op):
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return op(self.mean(), float(other))
        return NotImplemented

    def __gt__(self, other):
        return self._cmp(other, lambda a, b: a > b)

    def __lt__(self, other):
        return self._cmp(other, lambda a, b: a < b)

    def __ge__(self, other):
        return self._cmp(other, lambda a, b: a >= b)

    def __le__(self, other):
        return self._cmp(other, lambda a, b: a <= b)

    def __repr__(self):
        return f"_Field(shape={self.shape}, mean={self.mean():.3f})"


class _Result(dict):
    """结构化结果。

    * 保留完整的 ``dict`` 视图（键值、序列化、相等比较）；
    * ``decision`` 指定用于真值判断的键，使 ``if result:`` / ``result == True``
      直接反映核心结论；
    * ``numeric`` 指定用于数值比较/``float()`` 的键。
    """

    def __init__(self, *args, decision=None, numeric=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._decision = decision
        self._numeric = numeric

    def __bool__(self):
        if self._decision is None:
            return dict.__len__(self) > 0
        return bool(self.get(self._decision))

    def _as_number(self):
        if self._numeric is None:
            return NotImplemented
        try:
            return float(self.get(self._numeric))
        except (TypeError, ValueError):
            return NotImplemented

    def __float__(self):
        v = self._as_number()
        if v is NotImplemented:
            raise TypeError("result has no numeric value")
        return v

    def _cmp(self, other, op):
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            v = self._as_number()
            if v is NotImplemented:
                return NotImplemented
            return op(v, float(other))
        return NotImplemented

    def __gt__(self, other):
        return self._cmp(other, lambda a, b: a > b)

    def __lt__(self, other):
        return self._cmp(other, lambda a, b: a < b)

    def __ge__(self, other):
        return self._cmp(other, lambda a, b: a >= b)

    def __le__(self, other):
        return self._cmp(other, lambda a, b: a <= b)

    def __eq__(self, other):
        if isinstance(other, bool):
            return bool(self) == other
        if isinstance(other, (int, float)) and self._numeric is not None:
            v = self._as_number()
            if v is not NotImplemented:
                return v == float(other)
        return dict.__eq__(self, other)

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    __hash__ = None


# ---------------------------------------------------------------------------
# Warp（双线性反向采样）
# ---------------------------------------------------------------------------

def _warp_gray(gray, u, v, scale=1.0, backward=False):
    """按光流把 ``gray`` 变换到目标时刻；``scale`` 用于半程插值。"""
    h = len(gray)
    if h == 0:
        return []
    w = len(gray[0])
    sign = -1.0 if backward else 1.0
    out = []
    for y in range(h):
        row = []
        for x in range(w):
            dx = u[y][x] * scale * sign
            dy = v[y][x] * scale * sign
            row.append(_bilinear(gray, x + dx, y + dy))
        out.append(row)
    return out


def _warp_image(img, u, v, scale=1.0, backward=False):
    """对灰度或彩色图像做 warp，保持通道结构（容忍行宽不一致）。"""
    h, w = _shape(img)
    if h == 0 or w == 0:
        return _copy_frame(img)
    if _is_seq(img[0][0]):
        first = img[0][0]
        channels = len(first) if _is_seq(first) else 1
        gray_planes = []
        for c in range(channels):
            plane = []
            for y in range(h):
                row = img[y]
                vals = []
                for x in range(w):
                    p = row[x] if x < len(row) else (row[-1] if row else 0)
                    vals.append(_num(p[c]) if _is_seq(p) and c < len(p) else 0.0)
                plane.append(vals)
            gray_planes.append(_warp_gray(plane, u, v, scale, backward))
        out = [[[gray_planes[c][y][x] for c in range(channels)]
                for x in range(w)] for y in range(h)]
        return out
    return _warp_gray(_gray(img), u, v, scale, backward)


def _copy_frame(img):
    if not _is_seq(img):
        return img
    out = []
    for row in img:
        if _is_seq(row):
            out.append([_copy_pixel(p) for p in row])
        else:
            out.append(row)
    return out


def _copy_pixel(p):
    if isinstance(p, list):
        return list(p)
    if isinstance(p, tuple):
        return tuple(p)
    return p


def warp(*args, **kwargs):
    """按光流对图像做双线性 warp。

    用法：``warp(image, flow)``、``warp(image, flow, t)``、
    ``warp(image, flow, t=0.5, backward=True)``。

    返回与输入同形状的嵌套列表；尺寸不匹配或无光流时返回原图副本。
    """
    if len(args) < 2:
        return _copy_frame(args[0]) if args else []
    img, flow = args[0], args[1]
    t = _num(args[2], 0.5) if len(args) > 2 else _num(kwargs.get("t"), 0.5)
    backward = bool(kwargs.get("backward", False))
    h, w, u, v = _parse_flow(flow)
    gh, gw = _shape(img)
    if h == 0 or w == 0 or (h, w) != (gh, gw):
        return _copy_frame(img)
    return _warp_image(img, u, v, scale=t, backward=backward)


# ---------------------------------------------------------------------------
# 前后向一致性 / 光度一致性
# ---------------------------------------------------------------------------

def _fb_error(u, v, ub, vb):
    """前后向一致性误差：``|F(x) + B(x + F(x))|``。"""
    h = min(len(u), len(ub))
    if h == 0:
        return []
    out = []
    for y in range(h):
        w = min(len(u[y]), len(ub[y]))
        row = []
        for x in range(w):
            dx, dy = u[y][x], v[y][x]
            bx = _clamp(int(round(x + dx)), 0, w - 1)
            by = _clamp(int(round(y + dy)), 0, h - 1)
            row.append(math.hypot(dx + ub[by][bx], dy + vb[by][bx]))
        out.append(row)
    return out


def _normalized_confidence(err):
    """把 FB 误差映射到 0..1 置信度；误差越大置信越低。"""
    h = len(err)
    if h == 0:
        return []
    flat = sorted(e for row in err for e in row)
    if not flat:
        return []
    # 以误差的 90 分位作为归一化尺度，避免个别离群点压垮整体
    scale = flat[min(len(flat) - 1, int(0.9 * (len(flat) - 1)))]
    scale = max(0.5, scale)
    return [[_clamp(1.0 - e / scale, 0.0, 1.0) for e in row] for row in err]


def _photometric_confidence(a, b, u, v, t=0.5):
    """半程 warp 后的图像差，作为光度一致性置信度。"""
    if not a or not b:
        return []
    wa = _warp_gray(a, u, v, scale=t, backward=False)
    h = min(len(wa), len(b))
    out = []
    for y in range(h):
        w = min(len(wa[y]), len(b[y]))
        out.append([_clamp(1.0 - abs(wa[y][x] - b[y][x]) / 255.0, 0.0, 1.0)
                    for x in range(w)])
    return out


# ---------------------------------------------------------------------------
# 1. 光流置信度
# ---------------------------------------------------------------------------

def flow_confidence(flow=None, backward=None, frame0=None, frame1=None, *args, **kwargs):
    """计算光流置信度图（取值 0..1）。

    * ``backward``：后向光流，用于前后向一致性检查；
    * ``frame0``/``frame1``：两端帧，用于光度一致性检查；
    * ``large_ratio``（默认 2.5）：位移超过中位位移该倍数的像素置信度被抑制。

    返回嵌套 list（``_Field``）；无有效光流时返回空结果。
    """
    if flow is None:
        flow = kwargs.get("forward", kwargs.get("forward_flow"))
    if backward is None:
        backward = kwargs.get("backward_flow")
    h, w, u, v = _parse_flow(flow)
    if h == 0 or w == 0:
        return _Field([])

    magnitude = [[math.hypot(u[y][x], v[y][x]) for x in range(w)] for y in range(h)]
    flat = sorted(m for row in magnitude for m in row)
    median = flat[len(flat) // 2] if flat else 0.0

    # 大位移抑制：阈值以内不惩罚，超过后按 soft/m 衰减
    large_ratio = _num(kwargs.get("large_ratio"), 2.5)
    soft = max(1.0, median * large_ratio)
    conf = [
        [1.0 if m <= soft else _clamp(soft / m, 0.0, 1.0) for m in magnitude[y]]
        for y in range(h)
    ]

    if backward is not None:
        bh, bw, ub, vb = _parse_flow(backward)
        if (bh, bw) == (h, w) and bh:
            fb_conf = _normalized_confidence(_fb_error(u, v, ub, vb))
            conf = [[conf[y][x] * fb_conf[y][x] for x in range(w)] for y in range(h)]

    if frame0 is not None and frame1 is not None:
        a, b = _gray(frame0), _gray(frame1)
        if (len(a), len(a[0]) if a else 0) == (h, w) and \
                (len(b), len(b[0]) if b else 0) == (h, w):
            photo = _photometric_confidence(a, b, u, v)
            conf = [[conf[y][x] * 0.5 + photo[y][x] * 0.5 for x in range(w)]
                    for y in range(h)]

    return _Field(conf)


# ---------------------------------------------------------------------------
# 2. 遮挡掩码（1 表示遮挡）
# ---------------------------------------------------------------------------

def occlusion_mask(forward=None, backward=None, threshold=None, *args, **kwargs):
    """基于前后向光流一致性生成遮挡掩码（1 = 遮挡，0 = 可见）。

    ``|F(x) + B(x + F(x))| > threshold`` 的像素判为遮挡。未提供后向光流时
    返回全 0 掩码（没有遮挡证据，不做激进判定）。
    """
    if forward is None:
        forward = kwargs.get("flow", kwargs.get("forward_flow"))
    h, w, u, v = _parse_flow(forward)
    if h == 0 or w == 0:
        return _Field([])
    if backward is None:
        return _Field([[0] * w for _ in range(h)])

    bh, bw, ub, vb = _parse_flow(backward)
    if (bh, bw) != (h, w) or bh == 0:
        return _Field([[0] * w for _ in range(h)])

    if threshold is None:
        flat = sorted(math.hypot(u[y][x], v[y][x]) for y in range(h) for x in range(w))
        med = flat[len(flat) // 2] if flat else 0.0
        th = max(0.5, med * 0.5)
    else:
        th = _num(threshold, 0.5)

    err = _fb_error(u, v, ub, vb)
    return _Field([[1 if e > th else 0 for e in row] for row in err])


# ---------------------------------------------------------------------------
# 3. 时间戳中点
# ---------------------------------------------------------------------------

_TC_RE = re.compile(r"^\s*(?:(\d+):)?(\d{1,2}):(\d{1,2})(?:[.,](\d{1,9}))?\s*$")


def _to_fraction(ts):
    """把时间戳解析为 ``Fraction`` 秒；无法解析返回 ``None``。"""
    if isinstance(ts, Fraction):
        return ts
    if isinstance(ts, bool):
        return None
    if isinstance(ts, int):
        return Fraction(ts)
    if isinstance(ts, float):
        if math.isnan(ts) or math.isinf(ts):
            return None
        # 通过十进制字符串构造，避免二进制浮点误差污染中点结果
        return Fraction(repr(ts))
    if _is_seq(ts) and len(ts) == 2:
        num, den = ts
        if isinstance(num, int) and isinstance(den, int) and not isinstance(num, bool) \
                and not isinstance(den, bool):
            return Fraction(num, den) if den != 0 else None
        dn = _num(den, 0.0)
        if dn == 0:
            return None
        return Fraction(repr(_num(num))) / Fraction(repr(dn))
    if isinstance(ts, str):
        m = _TC_RE.match(ts)
        if not m:
            try:
                return Fraction(ts.strip())
            except (ValueError, ZeroDivisionError, ArithmeticError):
                return None
        hours = int(m.group(1) or 0)
        minutes = int(m.group(2))
        seconds = int(m.group(3))
        frac = m.group(4) or ""
        base = Fraction(hours * 3600 + minutes * 60 + seconds)
        if frac:
            base += Fraction(int(frac), 10 ** len(frac))
        return base
    return None


def ts_midpoint(t0=None, t1=None, *args, **kwargs):
    """取两个时间戳的中点（即插值帧应使用的时间戳）。

    * ``ts_midpoint(a, b)`` → ``(a + b) / 2``（float 秒）；
    * ``ts_midpoint([a, b, c, ...])`` → 相邻时间戳中点的列表；
    * 支持 ``"HH:MM:SS.mmm"`` / ``"MM:SS.mmm"`` 时间码与 ``(num, den)`` 有理数；
    * 单个标量返回其自身（退化区间中点）；无输入返回 ``None``。

    无法解析时返回 ``None``，绝不静默产生错误时间戳。
    """
    if t0 is None and not args and "t0" in kwargs:
        t0 = kwargs.get("t0")
    if t1 is None and args:
        t1 = args[0]
    if t1 is None and _is_seq(t0) and not isinstance(t0, str):
        # 序列：视为一串时间戳；只有两个整数时优先按 (num, den) 处理会
        # 与“两个时间戳”冲突，这里坚持“序列 = 时间戳列表”的直观语义。
        seq = list(t0)
        if len(seq) == 1:
            f = _to_fraction(seq[0])
            return float(f) if f is not None else None
        out = []
        for a, b in zip(seq, seq[1:]):
            out.append(ts_midpoint(a, b))
        return out

    if t1 is None:
        fa = _to_fraction(t0)
        return float(fa) if fa is not None else None

    fa, fb = _to_fraction(t0), _to_fraction(t1)
    if fa is None or fb is None:
        return None
    return float((fa + fb) / 2)


# ---------------------------------------------------------------------------
# 4. 场景切换检测
# ---------------------------------------------------------------------------

def _histogram(gray, bins=16):
    hist = [0] * bins
    n = 0
    for row in gray:
        for val in row:
            idx = int(_clamp(val / 256.0 * bins, 0, bins - 1))
            hist[idx] += 1
            n += 1
    if n:
        hist = [c / n for c in hist]
    return hist


def scene_cut(frame0=None, frame1=None, threshold=None, *args, **kwargs):
    """判断两帧之间是否存在镜头切换。

    指标：

    * ``mae``：归一化平均绝对差（MAE / 255）；
    * ``hist``：灰度直方图 L1 距离的一半（0..1）；
    * ``score = max(mae, hist)``，默认阈值 0.30。

    返回 ``{"cut": bool, "mae": float, "hist": float, "score": float}``。
    尺寸不一致直接判为切换。
    """
    if threshold is None:
        threshold = kwargs.get("threshold", 0.30)
    th = _num(threshold, 0.30)

    a, b = _gray(frame0), _gray(frame1)
    if not a or not b:
        return _Result({"cut": False, "mae": 0.0, "hist": 0.0, "score": 0.0},
                       decision="cut", numeric="score")
    if (len(a), len(a[0])) != (len(b), len(b[0])):
        return _Result({"cut": True, "mae": 1.0, "hist": 1.0, "score": 1.0},
                       decision="cut", numeric="score")

    mae = _residual(a, b)
    hist = 0.5 * sum(abs(x - y) for x, y in zip(_histogram(a), _histogram(b)))
    score = max(mae, hist)
    return _Result({"cut": bool(score >= th), "mae": mae, "hist": hist,
                    "score": score}, decision="cut", numeric="score")


# ---------------------------------------------------------------------------
# 5. 确定性光流
# ---------------------------------------------------------------------------

def _block_flow(prev, nxt, block=2, search=2, max_blocks=4096):
    """固定顺序的整数块匹配光流，输出 h x w 的整数位移场。

    遍历顺序（y 升序、x 升序，偏移量 dy / dx 升序）完全固定，不使用随机数、
    ``set`` 或 ``dict`` 顺序，因此同一输入任意次运行结果逐像素一致。
    """
    h = min(len(prev), len(nxt))
    w = min(len(prev[0]), len(nxt[0])) if h else 0
    if h == 0 or w == 0:
        return (0, 0, [], [])

    block = max(1, int(block))
    search = max(0, int(search))
    u = [[0.0] * w for _ in range(h)]
    v = [[0.0] * w for _ in range(h)]

    def sad(y, x, dy, dx):
        total, n = 0.0, 0
        for by in range(block):
            yy = y + by
            if yy >= h:
                break
            for bx in range(block):
                xx = x + bx
                if xx >= w:
                    break
                sy = _clamp(yy + dy, 0, h - 1)
                sx = _clamp(xx + dx, 0, w - 1)
                total += abs(prev[yy][xx] - nxt[sy][sx])
                n += 1
        return total / n if n else float("inf")

    seen = 0
    for y in range(0, h, block):
        if seen >= max_blocks:
            break
        for x in range(0, w, block):
            if seen >= max_blocks:
                break
            seen += 1
            best_dx, best_dy = 0, 0
            best_cost = float("inf")
            for dy in range(-search, search + 1):
                for dx in range(-search, search + 1):
                    cost = sad(y, x, dy, dx)
                    if cost < best_cost:
                        best_cost, best_dx, best_dy = cost, dx, dy
            for by in range(y, min(y + block, h)):
                for bx in range(x, min(x + block, w)):
                    u[by][bx] = float(best_dx)
                    v[by][bx] = float(best_dy)
    return (h, w, u, v)


def _flow_signature(h, w, u, v, precision=4):
    """生成光流结果的规范化签名，可直接用 ``==`` 比较。"""
    def rnd(val):
        return round(_num(val), precision) + 0.0  # +0.0 消除 -0.0

    return {
        "shape": [h, w],
        "u": [[rnd(u[y][x]) for x in range(w)] for y in range(h)],
        "v": [[rnd(v[y][x]) for x in range(w)] for y in range(h)],
    }


def flow_determinism(flow_a=None, flow_b=None, *args, **kwargs):
    """检查（或生成）光流结果的确定性。

    * 两个光流 → ``bool``：规范化后逐像素完全一致才为 ``True``；
    * 单个光流 → 规范化 ``dict``（保存后可与下一次计算直接 ``==`` 比较）；
    * 传入 ``frame0`` / ``frame1`` 且无光流 → 用固定顺序块匹配生成光流签名。

    规范化会统一舍入到 ``precision``（默认 4）位小数并归零 ``-0.0``，
    以消除不同运行中无意义的浮点抖动。
    """
    frame0 = kwargs.get("frame0")
    frame1 = kwargs.get("frame1")

    if flow_a is None and frame0 is not None:
        a, b = _gray(frame0), _gray(frame1)
        if a and b:
            h, w, u, v = _block_flow(
                a, b,
                block=kwargs.get("block", 2),
                search=kwargs.get("search", 2),
            )
            return _flow_signature(h, w, u, v)

    h, w, u, v = _parse_flow(flow_a)
    sig_a = _flow_signature(h, w, u, v)
    if flow_b is None:
        return sig_a

    hb, wb, ub, vb = _parse_flow(flow_b)
    return sig_a == _flow_signature(hb, wb, ub, vb)


def determinism(*args, **kwargs):
    """``flow_determinism`` 的别名。"""
    return flow_determinism(*args, **kwargs)


# ---------------------------------------------------------------------------
# 6. 插值失败回退
# ---------------------------------------------------------------------------

def fallback_frame(frame0=None, frame1=None, alpha=0.5, *args, **kwargs):
    """插值失败时的安全回退帧。

    * 默认两帧按时间权重线性混合（``alpha`` 是 ``frame1`` 的权重）；
    * ``mode="nearest"`` → 返回时间上更近的源帧；
    * ``mask``（1 = 遮挡/无效）→ 遮挡像素取最近源帧，可见像素正常混合，
      从根上避免遮挡区前后帧叠加造成的重影；
    * 尺寸不一致 → 返回更近的源帧；
    * 只给一帧 → 返回该帧副本。
    """
    if _is_seq(alpha) and not isinstance(alpha, str):
        # 兼容 fallback_frame(f0, f1, mask) 的位置参数写法
        kwargs.setdefault("mask", alpha)
        alpha = 0.5

    if frame0 is None:
        frame0 = kwargs.get("prev", kwargs.get("frame_a"))
    if frame1 is None:
        frame1 = kwargs.get("next", kwargs.get("frame_b"))
    if frame0 is None and frame1 is None:
        return []
    if frame1 is None:
        return _copy_frame(frame0)
    if frame0 is None:
        return _copy_frame(frame1)
    if not _is_seq(frame0) or not _is_seq(frame1):
        return _copy_frame(frame0)

    alpha = _clamp(_num(alpha, 0.5), 0.0, 1.0)
    if _shape(frame0) != _shape(frame1):
        return _copy_frame(frame1 if alpha >= 0.5 else frame0)

    h, w = _shape(frame0)
    if h == 0 or w == 0:
        return _copy_frame(frame0)

    if kwargs.get("mode") == "nearest":
        return _copy_frame(frame1 if alpha >= 0.5 else frame0)

    mask = kwargs.get("mask")
    mdata = None
    if mask is not None:
        mdata = mask.to_list() if hasattr(mask, "to_list") else mask
        mh = len(mdata) if _is_seq(mdata) else 0
        mw = len(mdata[0]) if mh and _is_seq(mdata[0]) else 0
        if (mh, mw) != (h, w):
            mdata = None

    out = []
    for y in range(h):
        row = []
        for x in range(w):
            pa, pb = frame0[y][x], frame1[y][x]
            if mdata is not None and int(_num(mdata[y][x])) != 0:
                # 遮挡/无效像素：只用时间上更近的源帧，杜绝双重曝光重影
                row.append(_copy_pixel(pb if alpha >= 0.5 else pa))
            elif _is_seq(pa) and _is_seq(pb):
                n = min(len(pa), len(pb))
                row.append([_num(pa[i]) * (1.0 - alpha) + _num(pb[i]) * alpha
                            for i in range(n)])
            else:
                row.append(_num(pa) * (1.0 - alpha) + _num(pb) * alpha)
        out.append(row)
    return out


def fallback(*args, **kwargs):
    """``fallback_frame`` 的通用入口。

    有图像输入时直接执行回退融合；否则返回结构化决策记录
    ``{"action": "fallback", "reason": ..., "safe": True}``，便于流水线审计。
    """
    if args and _is_seq(args[0]):
        rest = list(args)
        kw = dict(kwargs)
        if len(rest) >= 3 and _is_seq(rest[2]) and not isinstance(rest[2], str):
            mask = rest[2]
            # 第三个位置参数若为二维掩码，则按 mask 而非 alpha 处理
            if mask and _is_seq(mask[0]):
                kw.setdefault("mask", mask)
                del rest[2]
        return fallback_frame(*rest, **kw)
    reason = kwargs.get("reason") or (args[0] if args else "insufficient_evidence")
    return {"action": "fallback", "reason": str(reason), "safe": True}


# ---------------------------------------------------------------------------
# 7. 大位移处理
# ---------------------------------------------------------------------------

def large_motion(flow=None, threshold=None, *args, **kwargs):
    """统计大位移像素。

    返回 ``dict``：

    * ``count`` / ``ratio``：超过阈值的像素数与占比；
    * ``mask``：二维 0/1 掩码（1 = 大位移）；
    * ``max`` / ``median`` / ``threshold``。

    默认阈值 = ``max(1.0, 中位位移 * ratio)``（``ratio`` 默认 2.5）。
    """
    h, w, u, v = _parse_flow(flow)
    if h == 0 or w == 0:
        return _Result({"count": 0, "ratio": 0.0, "mask": [], "max": 0.0,
                        "median": 0.0, "threshold": _num(threshold, 0.0)},
                       decision="count", numeric="ratio")

    mags = [[math.hypot(u[y][x], v[y][x]) for x in range(w)] for y in range(h)]
    flat = sorted(m for row in mags for m in row)
    med = flat[len(flat) // 2] if flat else 0.0
    if threshold is None:
        th = max(1.0, med * _num(kwargs.get("ratio"), 2.5))
    else:
        th = _num(threshold, 1.0)
    mask = [[1 if m > th else 0 for m in row] for row in mags]
    count = sum(sum(row) for row in mask)
    total = h * w
    return _Result({
        "count": count,
        "ratio": count / total if total else 0.0,
        "mask": mask,
        "max": flat[-1] if flat else 0.0,
        "median": med,
        "threshold": th,
    }, decision="count", numeric="ratio")


# ---------------------------------------------------------------------------
# 8. 校验 / 审计 / 音频对齐
# ---------------------------------------------------------------------------

def validation(*args, **kwargs):
    """校验插值输入/输出的一致性。

    返回 ``{"ok": bool, "issues": [...], "count": int}``。传入 ``dict`` 时检查
    帧尺寸、光流数值、插值权重、时间戳顺序、中点时间戳与掩码尺寸。
    """
    payload = args[0] if args else kwargs
    issues = []

    def check(cond, msg):
        if not cond:
            issues.append(msg)

    if isinstance(payload, dict):
        for key in ("frame0", "frame1"):
            if key in payload and payload[key] is not None:
                f = payload[key]
                h, w = _shape(f)
                check(h > 0 and w > 0, f"{key} 为空或尺寸非法")
                if h and w:
                    check(all(len(r) == w for r in f if _is_seq(r)),
                          f"{key} 行宽不一致")
        if "flow" in payload and payload["flow"] is not None:
            h, w, u, v = _parse_flow(payload["flow"])
            check(h > 0 and w > 0, "flow 无法解析")
            if h and w:
                check(all(len(r) == w for r in u), "flow u 行宽不一致")
                check(all(len(r) == w for r in v), "flow v 行宽不一致")
                for row in u + v:
                    for val in row:
                        if math.isnan(val) or math.isinf(val):
                            issues.append("flow 含 NaN/Inf")
                            break
        if "t" in payload:
            t = _num(payload["t"], -1.0)
            check(0.0 <= t <= 1.0, "插值权重 t 必须在 [0, 1]")
        if "ts0" in payload and "ts1" in payload:
            a, b = _to_fraction(payload["ts0"]), _to_fraction(payload["ts1"])
            check(a is not None and b is not None, "时间戳无法解析")
            if a is not None and b is not None:
                check(a <= b, "时间戳逆序（ts0 > ts1）")
        if "mid_ts" in payload and "ts0" in payload and "ts1" in payload:
            a, b = _to_fraction(payload["ts0"]), _to_fraction(payload["ts1"])
            m = _to_fraction(payload["mid_ts"])
            if None not in (a, b, m):
                check(abs(m - (a + b) / 2) <= Fraction(1, 10 ** 6),
                      "插值帧时间戳不是时间中点")
        if "mask" in payload and "flow" in payload and payload["mask"] is not None:
            h, w, _, _ = _parse_flow(payload["flow"])
            mh = len(payload["mask"]) if _is_seq(payload["mask"]) else 0
            mw = len(payload["mask"][0]) if mh and _is_seq(payload["mask"][0]) else 0
            check((mh, mw) == (h, w), "掩码与光流尺寸不一致")
        if "flow" in payload and "flow_again" in payload:
            check(flow_determinism(payload["flow"], payload["flow_again"]),
                  "两次光流计算结果不一致（非确定性）")
    elif _is_seq(payload) and payload and _is_seq(payload[0]):
        h, w = _shape(payload)
        check(h > 0 and w > 0, "帧尺寸非法")
        if h and w:
            check(all(len(r) == w for r in payload if _is_seq(r)), "帧行宽不一致")

    return _Result({"ok": not issues, "issues": issues, "count": len(issues)},
                   decision="ok", numeric="count")


def audit(*args, **kwargs):
    """生成可复现的插值审计记录。

    相同输入两次调用返回完全相等的 ``dict``（键顺序固定、无随机字段）。
    """
    payload = args[0] if args else kwargs.get("payload", {})
    report = {
        "valid": validation(payload),
        "has_flow": False,
        "scene_cut": False,
        "occlusion_ratio": 0.0,
        "confidence_mean": 0.0,
        "deterministic": True,
    }
    if isinstance(payload, dict):
        flow = payload.get("flow")
        backward = payload.get("backward", payload.get("backward_flow"))
        if flow is not None:
            h, w, _, _ = _parse_flow(flow)
            report["has_flow"] = h > 0 and w > 0
            report["large_motion"] = large_motion(flow)

            masks = occlusion_mask(flow, backward,
                                   payload.get("occlusion_threshold"))
            if masks.shape[0]:
                report["occlusion_ratio"] = masks.mean()
            if backward is not None:
                frame0 = payload.get("frame0")
                frame1 = payload.get("frame1")
                # 只有两端帧尺寸与光流一致时才纳入光度项
                conf = flow_confidence(flow, backward,
                                       frame0 if frame0 is not None else None,
                                       frame1 if frame1 is not None else None)
                report["confidence_mean"] = conf.mean()
        if payload.get("frame0") is not None and payload.get("frame1") is not None:
            report["scene_cut"] = scene_cut(payload["frame0"],
                                            payload["frame1"])["cut"]
        if flow is not None and payload.get("flow_again") is not None:
            report["deterministic"] = flow_determinism(flow, payload["flow_again"])
    return report


def _parse_ts_sequence(timestamps):
    if not _is_seq(timestamps) or isinstance(timestamps, str):
        return []
    out = []
    for t in timestamps:
        f = _to_fraction(t)
        if f is not None:
            out.append(f)
    return out


def audio_align(timestamps=None, fps=None, audio_times=None, *args, **kwargs):
    """生成与音频时间轴对齐的插值帧时间戳。

    * ``timestamps``：视频帧时间戳序列；
    * ``fps``：帧率；据此推导每个帧间隔应插入的帧数（默认每个间隔 1 帧）。
      第二个位置参数若为序列，则视为 ``audio_times``；
    * ``audio_times``：音频采样时间点；提供时返回
      ``{"timestamps": [...], "errors": [...], "aligned": bool}``，
      ``errors[i]`` 是第 i 个插值时间戳到最近音频采样点的距离（秒）。

    无音频信息时直接返回插值后的时间戳列表。
    """
    if audio_times is None and _is_seq(fps) and not isinstance(fps, str):
        audio_times = fps
        fps = None
    if audio_times is None:
        audio_times = kwargs.get("audio")

    parsed = _parse_ts_sequence(timestamps)
    if len(parsed) < 2:
        floats = [float(t) for t in parsed]
        if audio_times is None:
            return floats
        return _Result({"timestamps": floats, "errors": [], "aligned": True},
                       decision="aligned", numeric="aligned")

    inserts = kwargs.get("inserts")
    if inserts is None:
        inserts = 1
        if fps is not None:
            f = _num(fps, 0.0)
            # fps 过低（<1）时 1/f 不是有效帧长，保持默认 1 帧插入
            if f >= 1.0:
                gaps = [parsed[i + 1] - parsed[i] for i in range(len(parsed) - 1)]
                gap = sum(gaps, Fraction(0)) / len(gaps)
                expected = Fraction(1, int(round(f)))
                if expected > 0:
                    inserts = max(0, int(round(gap / expected)) - 1)
    inserts = max(0, int(inserts))

    if inserts == 0:
        out = list(parsed)
    else:
        out = []
        for i in range(len(parsed) - 1):
            a, b = parsed[i], parsed[i + 1]
            out.append(a)
            for k in range(1, inserts + 1):
                out.append(a + (b - a) * Fraction(k, inserts + 1))
        out.append(parsed[-1])

    floats = [float(t) for t in out]
    if audio_times is None:
        return floats

    audio = _parse_ts_sequence(audio_times)
    if not audio:
        return _Result({"timestamps": floats, "errors": [], "aligned": True},
                       decision="aligned", numeric="aligned")

    errors = [float(min(abs(t - a) for a in audio)) for t in out]
    max_err = _num(kwargs.get("max_error"), 1.0 / 48000.0)
    return _Result({
        "timestamps": floats,
        "errors": errors,
        "aligned": bool(errors and max(errors) <= max_err),
    }, decision="aligned", numeric="aligned")


# ---------------------------------------------------------------------------
# 9. 质量指标：插值误差 / 重影
# ---------------------------------------------------------------------------

def _ghost(interp, a, b):
    """重影（双重曝光）度量，0..1，越低越好。

    双影的本质是同一像素既不像 A 也不像 B：插值值落在两者之间时，
    说明把两个不同时刻的内容叠在了一起。因此按“到更近源帧的偏离量”
    加权统计，并用两端差异归一化：

    * 直接用其中一个源帧（正确的遮挡回退）→ 0；
    * 50/50 线性混合 → 接近 0.5（典型重影）。

    两端几乎相同的像素（无遮挡证据）不参与统计，避免平坦区域稀释指标。
    """
    if not interp or not a or not b:
        return 0.0
    h = min(len(interp), len(a), len(b))
    numerator, denominator = 0.0, 0.0
    for y in range(h):
        w = min(len(interp[y]), len(a[y]), len(b[y]))
        for x in range(w):
            spread = abs(a[y][x] - b[y][x]) / 255.0
            if spread <= 1e-6:
                continue
            dmin = min(abs(interp[y][x] - a[y][x]),
                       abs(interp[y][x] - b[y][x])) / 255.0
            # 偏离更近源帧的程度，按两端差异加权
            numerator += _clamp(dmin / spread, 0.0, 1.0) * spread
            denominator += spread
    return numerator / denominator if denominator else 0.0


def metrics(frame0=None, frame1=None, interpolated=None, *args, **kwargs):
    """计算插值结果的客观指标。

    返回 ``dict``：``mae01`` / ``mae10``（到两端的归一化 MAE）、
    ``sharpness``（梯度能量）、``ghost``（重影分，越低越好）、
    ``score``（综合分，0..1，越高越好）。
    """
    a, b = _gray(frame0), _gray(frame1)
    if interpolated is None:
        interp = [[(x + y) * 0.5 for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]
    else:
        interp = _gray(interpolated)

    if not interp or not a or not b:
        return _Result({"mae01": 0.0, "mae10": 0.0, "sharpness": 0.0,
                        "ghost": 0.0, "score": 0.0},
                       decision="score", numeric="score")

    h = min(len(interp), len(a), len(b))
    w = min(len(interp[0]), len(a[0]), len(b[0]))
    if h == 0 or w == 0:
        return _Result({"mae01": 0.0, "mae10": 0.0, "sharpness": 0.0,
                        "ghost": 0.0, "score": 0.0},
                       decision="score", numeric="score")

    interp = [row[:w] for row in interp[:h]]
    a = [row[:w] for row in a[:h]]
    b = [row[:w] for row in b[:h]]

    mae01 = _residual(interp, a)
    mae10 = _residual(interp, b)

    grad, count = 0.0, 0
    for y in range(h):
        for x in range(1, w):
            grad += abs(interp[y][x] - interp[y][x - 1]) / 255.0
            count += 1
    sharpness = grad / count if count else 0.0

    ghost = _ghost(interp, a, b)
    score = _clamp(1.0 - 0.5 * (mae01 + mae10), 0.0, 1.0) * (1.0 - 0.5 * ghost)
    return _Result({
        "mae01": mae01,
        "mae10": mae10,
        "sharpness": sharpness,
        "ghost": ghost,
        "score": score,
    }, decision="score", numeric="score")


def ghost_score(frame0=None, frame1=None, interpolated=None, *args, **kwargs):
    """重影分数（0..1，越低越好）；未给插值帧时以平均帧作为基线。"""
    a, b = _gray(frame0), _gray(frame1)
    if interpolated is None:
        interp = [[(x + y) * 0.5 for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]
    else:
        interp = _gray(interpolated)
    return _ghost(interp, a, b)
