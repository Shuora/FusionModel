import numpy as np


def test_stacking_uses_oof_features():
    from src.fusion.stacking import build_meta_features

    pred_img = np.array([[0.7, 0.3], [0.2, 0.8]], dtype=float)
    pred_tls = np.array([[0.6, 0.4], [0.3, 0.7]], dtype=float)
    pred_fuse = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=float)

    meta = build_meta_features(pred_img, pred_tls, pred_fuse, folds=5)
    assert "entropy_fuse" in meta.columns
