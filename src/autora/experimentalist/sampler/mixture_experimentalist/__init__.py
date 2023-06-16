"""
Mixture Experimentalist Sampler
"""


import numpy as np
from typing import Optional


def adjust_distribution(p, temperature):
        # temperature cannot be 0
        assert temperature != 0, 'Temperature cannot be 0'

        #If the temperature is very low (close to 0), then the sampling will become almost deterministic, picking the event with the highest probability.
        #If the temperature is very high, then the sampling will be closer to uniform, with all events having roughly equal probability.
        
        p = p / np.sum(p)  # Normalizing the initial distribution
        p = np.exp(p / temperature)  
        final_p = p / np.sum(p) # Normalizing the final distribution
        return final_p



    
def mixture_sampler(condition_pool: np.ndarray, temperature: float, samplers: list, params: dict, num_samples: Optional[int] = None) -> np.ndarray:
    """

    Args:
        condition_pool: pool of experimental conditions to evaluate
        temperature: how random is selection of conditions (cannot be 0; (0:1) - the choices are more deterministic than the choices made wrt
        samplers: tuple containing sampler functions, their names, and weights 
        for sampler functions that return both positive and negative scores, user can provide a list with two weights: the first one will be applied to positive scores, the second one -- to the negative
        params: nested dictionary. keys correspond to the sampler function names (same as provided in samplers),
        values correspond to the dictionaries of function arguments (argument name: its value)
        num_samples: number of experimental conditions to select
        
    Returns:
        Sampled pool of experimental conditions
    """
    
    rankings = []
    scores = []
    
    ## getting rankings and weighted scores from each function
    for (function, name, weight) in samplers:
        sampler_params = params[name]
        cur_ranking, cur_scores = function(condition_pool=condition_pool, **sampler_params)
        cur_indices = np.argsort(cur_ranking, axis=None)
        cur_ranking_sorted = cur_ranking[cur_indices]
        rankings.append(cur_ranking_sorted) # for checking: all elements should be the same & same order
        ## if function scores can be negative, then create a reversed dimension for them
        if np.sum(cur_scores<0)>0:
            
            cur_scores_positive = np.copy(cur_scores)
            cur_scores_positive[cur_scores<0]=0
            cur_scores_negative = -np.copy(cur_scores)
            cur_scores_negative[cur_scores>0]=0
            
            # aligning scores
            cur_scores_positive_sorted = cur_scores_positive[cur_indices]
            cur_scores_negative_sorted = cur_scores_negative[cur_indices]
            
            # if only one weight is provided, use it for both negative and positive dimensions
            if isinstance(weight, int):
                cur_scores_positive_weighted = cur_scores_positive_sorted * weight
                cur_scores_negative_weighted = cur_scores_negative_sorted * weight
            else:
                cur_scores_positive_weighted = cur_scores_positive_sorted * weight[0] # positive dimension gets the first weight
                cur_scores_negative_weighted = cur_scores_negative_sorted * weight[1] # negative dimension gets the second weight
            
            scores.append(cur_scores_positive_weighted)
            scores.append(cur_scores_negative_weighted)
            
        else:
            cur_scores_sorted = cur_scores[cur_indices]
            if isinstance(weight, int):
                cur_scores_weighted = cur_scores_sorted * weight
            else: 
                cur_scores_weighted = cur_scores_sorted * weight[0]
            scores.append(cur_scores_weighted)
    
    weighted_mixture_scores = np.sum(scores, axis = 0)
    
    # adjust mixture scores wrt temperature
    weighted_mixture_scores_adjusted = adjust_distribution(weighted_mixture_scores, temperature)
    
    if num_samples is None:
        num_samples = condition_pool.shape[0]
    
    conditions = np.random.choice(cur_ranking_sorted.T.squeeze(), num_samples,
              p=weighted_mixture_scores_adjusted, replace = False)
    
    return conditions
