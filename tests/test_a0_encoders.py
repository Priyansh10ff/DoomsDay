"""Encoder shapes, set-encoder permutation invariance, empty set, padded garbage."""
import numpy as np
import pytest
import torch

import a0_helpers  # noqa: F401  (thread cap, sys.path)
from agent.encoders import EntitySetEncoder, PrevActionEncoder, RayEncoder, SelfEncoder
from configs.model_config import ModelConfig

CFG = ModelConfig()


def make_set_encoder(n_types: int = 14, seed: int = 0) -> EntitySetEncoder:
    torch.manual_seed(seed)
    enc = EntitySetEncoder(8, n_types, CFG)
    return enc.eval()


def random_set(b: int, n: int, n_types: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(b, n, 8, generator=g)
    types = torch.randint(1, n_types, (b, n), generator=g)
    mask = torch.rand(b, n, generator=g) < 0.6
    return feats, types, mask


@pytest.mark.parametrize("bands,width", [(3, 80), (4, 80), (3, 16)])
def test_ray_encoder_shape(bands: int, width: int) -> None:
    enc = RayEncoder(bands, width, CFG)
    assert enc(torch.rand(5, bands, width)).shape == (5, CFG.ray_dim)


def test_self_and_prev_action_shapes() -> None:
    assert SelfEncoder(28, CFG)(torch.rand(7, 28)).shape == (7, CFG.self_dim)
    pa = PrevActionEncoder((3, 3, 7, 2, 2, 2), CFG)
    assert pa(torch.zeros(7, 6, dtype=torch.long)).shape == (7, CFG.prev_action_dim)


@pytest.mark.parametrize("n", [1, 5, 24])
def test_set_encoder_permutation_invariant(n: int) -> None:
    enc = make_set_encoder()
    feats, types, mask = random_set(6, n, 14, seed=n)
    out = enc(feats, types, mask)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(99))
    out_p = enc(feats[:, perm], types[:, perm], mask[:, perm])
    assert out.shape == (6, CFG.entity_token_dim)
    torch.testing.assert_close(out, out_p, atol=1e-5, rtol=1e-5)


def test_set_encoder_permutation_invariant_in_train_mode() -> None:
    enc = make_set_encoder().train()
    feats, types, mask = random_set(4, 24, 14, seed=3)
    perm = torch.arange(23, -1, -1)
    torch.testing.assert_close(enc(feats, types, mask), enc(feats[:, perm], types[:, perm], mask[:, perm]),
                               atol=1e-5, rtol=1e-5)


def test_set_encoder_empty_set_is_finite_and_constant() -> None:
    enc = make_set_encoder()
    feats, types, _ = random_set(3, 24, 14, seed=1)
    empty = torch.zeros(3, 24, dtype=torch.bool)
    out = enc(feats, types, empty)
    assert torch.isfinite(out).all()
    # with no visible entity the content of the rows must not matter
    out2 = enc(torch.randn(3, 24, 8) * 100, torch.randint(0, 14, (3, 24)), empty)
    torch.testing.assert_close(out, out2)
    torch.testing.assert_close(out[0], out[1])


def test_set_encoder_ignores_garbage_in_padded_rows() -> None:
    enc = make_set_encoder()
    feats, types, mask = random_set(4, 24, 14, seed=2)
    out = enc(feats, types, mask)
    g_feats = feats.clone()
    g_feats[~mask] = float("nan")
    g_feats[~mask] = torch.where(torch.rand_like(g_feats[~mask]) < 0.5, float("nan"), 1e6)
    g_types = types.clone()
    g_types[~mask] = 13
    out_g = enc(g_feats, g_types, mask)
    assert torch.isfinite(out_g).all()
    torch.testing.assert_close(out, out_g)


def test_set_encoder_any_count_and_row_position() -> None:
    """Same visible set placed in different rows of a padded array gives the same code."""
    enc = make_set_encoder()
    feats, types, _ = random_set(1, 3, 14, seed=4)
    a_f, a_t, a_m = torch.zeros(1, 24, 8), torch.zeros(1, 24, dtype=torch.long), torch.zeros(1, 24, dtype=torch.bool)
    b_f, b_t, b_m = a_f.clone(), a_t.clone(), a_m.clone()
    a_f[0, :3], a_t[0, :3], a_m[0, :3] = feats[0], types[0], True
    rows = torch.tensor([20, 2, 11])
    b_f[0, rows], b_t[0, rows], b_m[0, rows] = feats[0], types[0], True
    torch.testing.assert_close(enc(a_f, a_t, a_m), enc(b_f, b_t, b_m), atol=1e-5, rtol=1e-5)


def test_set_encoder_type_table_size_follows_argument() -> None:
    assert make_set_encoder(n_types=13).type_embed.num_embeddings == 13
    assert make_set_encoder(n_types=14).type_embed.num_embeddings == 14
    with pytest.raises(IndexError):
        enc = make_set_encoder(n_types=13)
        enc(torch.zeros(1, 2, 8), torch.tensor([[13, 0]]), torch.tensor([[True, False]]))


def test_set_encoder_deterministic_under_seed() -> None:
    feats, types, mask = random_set(2, 24, 14, seed=5)
    a = make_set_encoder(seed=7)(feats, types, mask)
    b = make_set_encoder(seed=7)(feats, types, mask)
    assert torch.equal(a, b)
    assert not np.allclose(a.detach().numpy(), make_set_encoder(seed=8)(feats, types, mask).detach().numpy())
