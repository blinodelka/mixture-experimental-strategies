"""
Example Experimentalist Sampler
"""


import numpy as np
from typing import Optional
from autora.experimentalist.sampler import get_scored_samples_from_model_prediction # not sure if the path is right
from autora.experimentalist.sampler import compute_dissimilarity # not sure if the path is right


def adjust_distribution(p, temperature):
        # temperature cannot be 0
        assert temperature != 0, 'Temperature cannot be 0'

        #If the temperature is very low (close to 0), then the sampling will become almost deterministic, picking the event with the highest probability.
        #If the temperature is very high, then the sampling will be closer to uniform, with all events having roughly equal probability.
        
        p = p / np.sum(p)  # Normalizing the initial distribution
        p = np.exp(p / temperature)  
        final_p = p / np.sum(p) # Normalizing the final distribution
        return final_p


    
def mixture_sampler(
    condition_pool: np.ndarray, weights: np.ndarray, temperature: int, 
    X_ref: np.ndarray, 
    X_train: np.ndarray,
    Y_train: np.ndarray, Y_predicted, num_samples: Optional[int] = None) -> np.ndarray:
    """
    Add a description of the sampler here.

    Args:
        condition_pool: pool of experimental conditions to evaluate
        num_samples: number of experimental conditions to select
        weights: array containing 4 weights -- importance of the falsification, confirmation, novelty, and familiarity (ideally, each pair of opposites? sums up to 1 or all? sum up to 1)
        temperature: how random is selection of conditions (cannot be 0; (0:1) - the choices are more deterministic than the choices made wrt
        the mixture scores; 1 - choices are made wrt to the mixture scores; (1, inf) - the choices are more random)
        X_ref, X_train, Y_train, Y_predicted: parameters required for falsification and novelty samplers
    
    Returns:
        Sampled pool of experimental conditions
    """
    
    
    falsification_ranking, falsification_scores = get_scored_samples_from_model_prediction(condition_pool, 
                                                                                           Y_predicted, X_train,
                                                                                           Y_train, n=condition_pool.shape[0])
    
    # getting rid of negative scores by introducing confirmation scores 
    confirmation_scores = -falsification_scores
    confirmation_scores[falsification_scores>0]=0
    falsification_scores[falsification_scores<0]=0
    
    # getting rid of negative scores by introducing familiarity scores 
    novelty_ranking, novelty_scores = compute_dissimilarity(condition_pool, X_ref, n=condition_pool.shape[0])
    
    familiarity_scores = -novelty_scores
    familiarity_scores[novelty_scores>0]=0
    novelty_scores[novelty_scores<0]=0
    
    # aligning the arrays based on the observations (condition pools)
    novelty_indices = np.argsort(novelty_ranking, axis=None)
    ranking_sorted = novelty_ranking[novelty_indices]
    novelty_scores_sorted = novelty_scores[novelty_indices]
    familiarity_scores_sorted = familiarity_scores[novelty_indices]

    falsification_indices = np.argsort(falsification_ranking, axis=None)
    falsification_scores_sorted = falsification_scores[falsification_indices]    
    confirmation_scores_sorted = confirmation_scores[falsification_indices] 
    
    weighted_mixture_scores = falsification_scores_sorted * weights[0] + confirmation_scores_sorted * weights[1] + novelty_scores_sorted * weights[2] + familiarity_scores_sorted * weights[3] 
    # each score is weighted by the relative importance of these different axes
    
    
    # adjust mixture scores wrt temperature
    weighted_mixture_scores_adjusted = adjust_distribution(weighted_mixture_scores, temperature)
    
    if num_samples is None:
        num_samples = condition_pool.shape[0]
    
    conditions = np.random.choice(ranking_sorted.T.squeeze(), num_samples,
              p=weighted_mixture_scores_adjusted, replace = False)
    
    return conditions
