import numpy as np
import pytest

torch = pytest.importorskip("torch")

from swipe_typing.layout import ALPHABET, KeyboardLayout   # noqa: E402
from swipe_typing.model.rescore import (                    # noqa: E402
    MAX_CAND_LEN,
    PAD,
    RescoreConfig,
    Rescorer,
    encode_candidates,
    listwise_loss,
)


# --- candidate encoding -----------------------------------------------------

def test_encode_candidates_shapes():
    ids, pos, mask = encode_candidates(["cat", "hello"])
    assert ids.shape == (2, MAX_CAND_LEN)
    assert pos.shape == (2, MAX_CAND_LEN, 2)
    assert mask.shape == (2, MAX_CAND_LEN)


def test_encode_candidates_positions_match_layout():
    kb = KeyboardLayout.qwerty()
    ids, pos, mask = encode_candidates(["cat"])
    for j, ch in enumerate("cat"):
        assert ids[0, j] == ALPHABET.index(ch)
        assert pos[0, j] == pytest.approx(kb.center(ch), abs=1e-5)
        assert not mask[0, j]
    assert mask[0, 3]                      # padded beyond the word
    assert ids[0, 3] == PAD


def test_encode_candidates_truncates_long_words():
    ids, _, mask = encode_candidates(["a" * 50])
    assert ids.shape[1] == MAX_CAND_LEN
    assert not mask[0].all()


def test_encode_candidates_empty_word_keeps_one_unmasked_slot():
    """A fully-masked attention row produces NaNs, so empty candidates must
    still expose one position."""
    _, _, mask = encode_candidates([""])
    assert not mask[0, 0]
    assert mask[0, 1:].all()


# --- model ------------------------------------------------------------------

def _batch(b=3, k=4, t=64, c=32):
    words = ["cat", "car", "hello", ""][:k] * b
    ids, pos, mask = encode_candidates(words)
    return (torch.randn(b, t, c),
            torch.from_numpy(ids.reshape(b, k, MAX_CAND_LEN)),
            torch.from_numpy(pos.reshape(b, k, MAX_CAND_LEN, 2)),
            torch.from_numpy(mask.reshape(b, k, MAX_CAND_LEN)),
            torch.randn(b, k))


def test_forward_shape_and_finiteness():
    m = Rescorer(RescoreConfig(d_model=32, dilations=(1, 2)))
    out = m(*_batch())
    assert out.shape == (3, 4)
    assert torch.isfinite(out).all()


def test_starts_as_a_refinement_of_the_first_pass():
    """first_pass_scale is initialised to 1 so training begins from the
    existing ranking rather than from noise."""
    m = Rescorer(RescoreConfig(d_model=32, dilations=(1,)))
    assert float(m.first_pass_scale.detach()) == pytest.approx(1.0)


def test_first_pass_score_moves_the_output():
    m = Rescorer(RescoreConfig(d_model=32, dilations=(1,)))
    m.eval()
    g, ids, pos, mask, fp = _batch()
    with torch.no_grad():
        a = m(g, ids, pos, mask, fp)
        b = m(g, ids, pos, mask, fp + 5.0)
    assert torch.allclose(b - a, torch.full_like(a, 5.0), atol=1e-4)


def test_candidates_are_scored_independently_of_their_slot():
    """Swapping two candidates must swap their scores, not change them."""
    m = Rescorer(RescoreConfig(d_model=32, dilations=(1,)))
    m.eval()
    ids, pos, mask = encode_candidates(["cat", "dog"])
    g = torch.randn(1, 64, 32)
    fp = torch.zeros(1, 2)
    ids_t = torch.from_numpy(ids)[None]
    pos_t = torch.from_numpy(pos)[None]
    mask_t = torch.from_numpy(mask)[None]
    with torch.no_grad():
        a = m(g, ids_t, pos_t, mask_t, fp)
        b = m(g, ids_t.flip(1), pos_t.flip(1), mask_t.flip(1), fp)
    assert torch.allclose(a, b.flip(1), atol=1e-4)


def test_gradients_flow():
    m = Rescorer(RescoreConfig(d_model=32, dilations=(1,)))
    out = m(*_batch())
    listwise_loss(out, torch.tensor([0, 1, -1]),
                  torch.ones(3, 4, dtype=torch.bool)).backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in m.parameters())


def test_normalization_round_trips():
    m = Rescorer(RescoreConfig(d_model=16, dilations=(1,)))
    m.set_normalization(np.arange(32), np.full(32, 2.0))
    m2 = Rescorer(RescoreConfig(d_model=16, dilations=(1,)))
    m2.load_state_dict(m.state_dict())
    assert torch.allclose(m.input_mean, m2.input_mean)
    m.set_normalization(np.zeros(32), np.zeros(32))
    assert (m.input_std >= 1e-3).all()


# --- loss -------------------------------------------------------------------

def test_listwise_loss_skips_lists_without_the_true_word():
    logits = torch.randn(4, 5, requires_grad=True)
    valid = torch.ones(4, 5, dtype=torch.bool)
    all_missing = listwise_loss(logits, torch.full((4,), -1), valid)
    assert float(all_missing) == 0.0


def test_listwise_loss_rewards_the_target():
    valid = torch.ones(2, 3, dtype=torch.bool)
    target = torch.tensor([0, 0])
    good = torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    bad = torch.tensor([[0.0, 10.0, 0.0], [0.0, 10.0, 0.0]])
    assert listwise_loss(good, target, valid) < listwise_loss(bad, target, valid)


def test_listwise_loss_masks_invalid_slots():
    """An empty candidate slot must never be selectable."""
    logits = torch.tensor([[0.0, 100.0]])
    valid = torch.tensor([[True, False]])
    loss = listwise_loss(logits, torch.tensor([0]), valid)
    assert float(loss) < 1e-3
