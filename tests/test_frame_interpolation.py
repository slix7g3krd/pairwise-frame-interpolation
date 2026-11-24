import pytest
import media.frame_interpolation as mod

CASES = [
    ("flow_confidence", (1,)),
    ("occlusion_mask", (1,)),
    ("ts_midpoint", (1,)),
    ("scene_cut", (1,)),
    ("flow_determinism", (1,)),
    ("fallback_frame", (1,)),
    ("large_motion", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("audio_align", (1,)),
    ("metrics", (1,)),
    ("ghost_score", (1,)),
    ("warp", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
