from autora.experimentalist.sampler.mixture_experimentalist import mixture_sampler
import numpy as np
import pytest

def mock_sampler(condition_pool, **kwargs):
    return condition_pool, np.random.rand(len(condition_pool))

def test_mixture_sampler():
    condition_pool = np.array([1, 2, 3, 4, 5])
    temperature = 0.5
    samplers = [[mock_sampler, "mock", 1]]
    params = {"mock": {}}

    # Test that the function runs without errors
    conditions = mixture_sampler(condition_pool, temperature, samplers, params)
    assert conditions is not None

    # Test that the function returns the correct number of samples
    conditions = mixture_sampler(condition_pool, temperature, samplers, params, num_samples=2)
    assert len(conditions) == 2

    # Test that the function returns unique samples when replace=False
    conditions = mixture_sampler(condition_pool, temperature, samplers, params, num_samples=len(condition_pool))
    assert len(conditions) == len(set(conditions))

    # Test that the function raises an error when temperature is 0
    with pytest.raises(AssertionError):
        conditions = mixture_sampler(condition_pool, 0, samplers, params)

    # Test that the function raises an error when num_samples is greater than the size of the condition pool
    with pytest.raises(ValueError):
        conditions = mixture_sampler(condition_pool, temperature, samplers, params, num_samples=len(condition_pool) + 1)

    # Test that the function raises an error when a sampler is not provided in the samplers list
    with pytest.raises(KeyError):
        conditions = mixture_sampler(condition_pool, temperature, [[mock_sampler, "nonexistent", 1]], params)