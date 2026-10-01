import numpy as np
import pytest
import torch

from statarb.utils import get_device, set_seed


def test_set_seed_reproduces_draws():
    set_seed(123)
    first = (np.random.rand(3), torch.rand(3))
    set_seed(123)
    second = (np.random.rand(3), torch.rand(3))
    assert np.array_equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_get_device():
    assert get_device("cpu").type == "cpu"
    assert get_device("auto").type in {"cpu", "cuda"}
    with pytest.raises(ValueError):
        get_device("tpu")
