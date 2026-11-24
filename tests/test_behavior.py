"""针对六个故障点的行为回归测试。"""

import media.frame_interpolation as fi


def _solid(h, w, value):
    return [[value] * w for _ in range(h)]


def _flow(h, w, dx=0.0, dy=0.0):
    return {"u": [[float(dx)] * w for _ in range(h)],
            "v": [[float(dy)] * w for _ in range(h)]}


# 1. 光流置信度 ------------------------------------------------------------

def test_confidence_rewards_fb_consistency():
    fwd = _flow(4, 4, 1.0, 0.0)
    good = fi.flow_confidence(fwd, _flow(4, 4, -1.0, 0.0))
    bad = fi.flow_confidence(fwd, _flow(4, 4, 0.0, 0.0))
    assert good.mean() > 0.9
    assert bad.mean() < 0.1


def test_confidence_suppresses_large_motion_outlier():
    h, w = 8, 8
    u = [[1.0] * w for _ in range(h)]
    u[3][3] = 40.0
    conf = fi.flow_confidence({"u": u, "v": [[0.0] * w for _ in range(h)]})
    assert conf[0][0] == 1.0
    assert conf[3][3] < 0.5


# 2. 遮挡掩码 --------------------------------------------------------------

def test_occlusion_mask_marks_only_inconsistent_pixels():
    fwd = _flow(3, 3, 1.0, 0.0)
    bwd = _flow(3, 3, -1.0, 0.0)
    bwd["u"][0][1] = 0.0  # 破坏像素 (0,0) 的前后向一致性
    mask = fi.occlusion_mask(fwd, bwd)
    assert mask[0][0] == 1
    assert mask[1][1] == 0


def test_occlusion_mask_without_backward_is_empty():
    assert fi.occlusion_mask(_flow(2, 2, 3.0, 3.0)).to_list() == [[0, 0], [0, 0]]


# 3. 时间戳中点 ------------------------------------------------------------

def test_ts_midpoint_exact_and_formats():
    assert fi.ts_midpoint(10.0, 10.04) == 10.02
    assert fi.ts_midpoint("00:00:01.000", "00:00:01.500") == 1.25
    assert fi.ts_midpoint((1, 2), (3, 2)) == 1.0
    assert fi.ts_midpoint([1.0, 2.0, 3.0]) == [1.5, 2.5]
    assert fi.ts_midpoint("not-a-time") is None


def test_audio_align_lands_on_audio_grid():
    grid = [i / 100 for i in range(9)]
    result = fi.audio_align([0.0, 0.04, 0.08], fps=50, audio_times=grid)
    assert result["aligned"] is True
    assert result["timestamps"] == [0.0, 0.02, 0.04, 0.06, 0.08]
    assert max(result["errors"]) == 0.0


# 4. 场景切换检测 ----------------------------------------------------------

def test_scene_cut_separates_similar_from_new_shot():
    a = _solid(8, 8, 100)
    assert fi.scene_cut(a, _solid(8, 8, 102))["cut"] is False
    assert fi.scene_cut(a, _solid(8, 8, 240))["cut"] is True


def test_scene_cut_mismatched_size_is_cut():
    assert fi.scene_cut(_solid(4, 4, 10), _solid(5, 5, 10))["cut"] is True


# 5. 确定性光流 ------------------------------------------------------------

def test_flow_determinism_same_input_same_output():
    a = [[(x * 7 + y * 3) % 256 for x in range(8)] for y in range(8)]
    b = fi.warp(a, _flow(8, 8, 1.0, 0.0), 1.0)
    assert fi.flow_determinism(None, frame0=a, frame1=b) == \
        fi.flow_determinism(None, frame0=a, frame1=b)


def test_flow_determinism_distinguishes_different_flows():
    assert fi.flow_determinism(_flow(4, 4, 1.0), _flow(4, 4, 1.0)) is True
    assert fi.flow_determinism(_flow(4, 4, 1.0), _flow(4, 4, 2.0)) is False


# 6. 插值失败回退 ----------------------------------------------------------

def test_fallback_uses_nearest_frame_in_occlusion():
    f0 = _solid(2, 2, 0)
    f1 = _solid(2, 2, 200)
    mask = [[1, 0], [0, 0]]
    out = fi.fallback_frame(f0, f1, 0.5, mask=mask)
    assert out[0][0] == 200          # 遮挡像素不复用前帧，避免重影
    assert out[0][1] == 100.0        # 可见像素正常混合


def test_fallback_reduces_ghosting_metric():
    f0 = _solid(6, 6, 0)
    f1 = _solid(6, 6, 200)
    mask = [[1, 1, 1, 0, 0, 0] for _ in range(6)]
    naive = fi.fallback_frame(f0, f1, 0.5)
    guarded = fi.fallback_frame(f0, f1, 0.5, mask=mask)
    assert fi.ghost_score(f0, f1, guarded) < fi.ghost_score(f0, f1, naive)


def test_fallback_nearest_mode_and_mismatched_size():
    assert fi.fallback_frame(_solid(2, 2, 5), _solid(2, 2, 50),
                             alpha=0.9, mode="nearest") == _solid(2, 2, 50)
    assert fi.fallback_frame(_solid(2, 2, 5), _solid(3, 3, 50),
                             alpha=0.9) == _solid(3, 3, 50)


# 综合 ---------------------------------------------------------------------

def test_audit_is_reproducible_and_validates():
    payload = {
        "frame0": _solid(6, 6, 10),
        "frame1": _solid(6, 6, 12),
        "flow": _flow(6, 6, 1.0),
        "backward": _flow(6, 6, -1.0),
        "ts0": 0.0, "ts1": 0.04, "mid_ts": 0.02, "t": 0.5,
    }
    first = fi.audit(payload)
    assert first == fi.audit(payload)
    assert first["valid"]["ok"] is True
    assert first["deterministic"] is True


def test_validation_catches_timestamp_and_weight_errors():
    bad = fi.validation({"ts0": 2.0, "ts1": 1.0, "mid_ts": 1.4, "t": 1.5})
    assert bad["ok"] is False
    assert bad["count"] == 3
