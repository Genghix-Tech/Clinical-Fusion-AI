import numpy as np
import torch

from clinicalfusion.models.encoders import ImageEncoder, UnimodalNet, build_ts_encoder
from clinicalfusion.models.fusion import AttentionFusion, LateFusionStacker, enable_mc_dropout


def _batch(b=6):
    return torch.randn(b, 1, 64, 64), torch.randn(b, 24), torch.randn(b, 4, 48)


def test_fusion_output_shapes_and_no_nan_even_with_no_modalities():
    m = AttentionFusion(24, 4).eval()
    img, tab, ts = _batch()
    for avail in (torch.ones(6, 3, dtype=torch.bool), torch.zeros(6, 3, dtype=torch.bool)):
        logit, w = m(img, tab, ts, avail)
        assert logit.shape == (6,) and w.shape == (6, 4)
        assert torch.isfinite(logit).all() and torch.allclose(w.sum(1), torch.ones(6), atol=1e-5)


def test_missing_modality_does_not_influence_output():
    m = AttentionFusion(24, 4).eval()
    img, tab, ts = _batch()
    avail = torch.tensor([[True, False, True]] * 6)
    a = m(img, tab, ts, avail)[0]
    b = m(img, torch.randn_like(tab) * 50, ts, avail)[0]  # garbage in the masked modality
    assert torch.allclose(a, b, atol=1e-5)
    assert (m(img, tab, ts, avail)[1][:, 1] == 0).all()  # zero attention on the masked modality


def test_mc_dropout_is_stochastic_but_eval_is_deterministic():
    m = AttentionFusion(24, 4, p=0.5)
    img, tab, ts = _batch(4)
    av = torch.ones(4, 3, dtype=torch.bool)
    m.eval()
    assert torch.equal(m(img, tab, ts, av)[0], m(img, tab, ts, av)[0])
    enable_mc_dropout(m)
    assert not torch.equal(m(img, tab, ts, av)[0], m(img, tab, ts, av)[0])
    assert not m.img.features[0][1].training  # BatchNorm stays in eval mode


def test_modality_dropout_only_active_in_training():
    m = AttentionFusion(24, 4, modality_dropout=1.0)  # drop everything while training
    img, tab, ts = _batch(4)
    av = torch.ones(4, 3, dtype=torch.bool)
    m.train()
    assert (m(img, tab, ts, av)[1][:, :3] == 0).all()
    m.eval()
    assert (m(img, tab, ts, av)[1][:, :3] > 0).all()


def test_ts_encoders_and_unimodal_net():
    for arch in ("tcn", "lstm"):
        net = UnimodalNet(build_ts_encoder(arch, 4, 64)).eval()
        assert net(torch.randn(5, 4, 48)).shape == (5,)
    assert UnimodalNet(ImageEncoder(64)).eval()(torch.randn(3, 1, 64, 64)).shape == (3,)


def test_stacker_ignores_missing_logits():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 600)
    L = np.stack([y + rng.normal(size=600), rng.normal(size=600), y + rng.normal(size=600)], 1)
    av = rng.random((600, 3)) < 0.7
    L[~av] = np.nan
    s = LateFusionStacker().fit(L, av, y)
    d = s.decision(L, av)
    assert np.isfinite(d).all()
    L2 = L.copy(); L2[~av] = 123.0  # whatever sits in missing slots must not matter
    assert np.allclose(d, s.decision(L2, av))
