# AutoRA Mixture Experimentalist

The Mixture Experimentalist identifies novel experimental conditions under which a hybrid of different experimental sampling strategies is used. 
This mixture can include any custom strategies such as falsification, novelty, crucial experimentation, uncertainty, elimination, aesthetic preferences, and arbitrary preferred/dispreferred regions of the space. The selection of conditions is based on a weighted sum of the scores obtained from these strategies.
 

## Quickstart Guide

You will need:

- Python 3.8 or greater: https://www.python.org/downloads/

Mixture Experimentalist is a part of the AutoRA package:

```bash
pip install -U autora["mixture_experimentalist"]
```

Check your installation by running:

```bash
python -c "from autora.experimentalist.sampler.mixture_experimentalist import mixture_sampler"
```

## Usage

The Mixture Experimentalist can be used to select experimental conditions based on a mixture of different strategies. Here's a basic example:

```python
from autora.experimentalist.sampler.mixture import mixture_sampler

# Define your condition pool, temperature, samplers, and parameters
condition_pool = ...
temperature = ...
samplers = ...
params = ...

# Use the mixture_sampler to select conditions
selected_conditions = mixture_sampler(
    condition_pool=condition_pool,
    temperature=temperature,
    samplers=samplers,
    params=params,
    num_samples=10
)
```
In this example, condition_pool is the pool of experimental conditions to evaluate, temperature controls the randomness of the selection (close to 0 for deterministic, higher for more random), samplers is a list of sampler functions with their weights in the mixture, and params is a dictionary of parameters for the sampler functions.

For more detailed usage instructions and examples, please refer to the documentation: https://github.com/blinodelka/mixture_experimental_strategies/blob/main/docs/basic-usage.ipynb.


