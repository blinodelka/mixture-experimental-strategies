from src.autora.experimentalist.mixture import mixture_sample
import numpy as np
import pytest
import pandas as pd

def mock_sampler(conditions, **kwargs):
    conditions["score"] = 1
    return conditions

def test_mixture_sample():
    condition_pool = pd.DataFrame({"x1": np.linspace(-3, 3, 7), "x2": np.linspace(-3, 3, 7)})

    temperature = 0.5
    samplers = [[mock_sampler, "mock", 1]]
    params = {"mock": {}}

    # Test that the function runs without errors
    conditions = mixture_sample(condition_pool, temperature, samplers, params)
    assert conditions is not None

    # Test that the function returns the correct number of samples
    conditions = mixture_sample(condition_pool, temperature, samplers, params, num_samples=2)
    assert len(conditions) == 2

    # Test that the function raises an error when temperature is 0
    with pytest.raises(AssertionError):
        conditions = mixture_sample(condition_pool, 0, samplers, params)

    # Test that the function raises an error when num_samples is greater than the size of the condition pool
    with pytest.raises(ValueError):
        conditions = mixture_sample(condition_pool, temperature, samplers, params, num_samples=len(condition_pool) + 1)
