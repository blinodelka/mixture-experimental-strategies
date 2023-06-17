# Tests

## Seeding

Some testcases involve random numbers. In order to avoid the testcases from running correctly sometimes, and incorrectly other times, we seed all the relevant random number generators for those testcases. To accomplish this, add the following pytest fixture to the test file and include it as required in the test functions:

```python
import random
import pytest
import torch

@pytest.fixture
def seed():
    """
    Ensures that the results are the same each time the tests are run.
    """
    random.seed(180)  # required for models which use the python `random` module
    torch.manual_seed(180)  # required for PyTorch models
    return


def test_foo(seed):
    """ Test something. """
    
    # No need to use `seed` in the function body – adding it as an argument is sufficient
    
    ... # Run tests
```

