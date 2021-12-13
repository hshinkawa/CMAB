import numpy as np
import itertools


def ModifiedSoftmax(result):
    """
    Input:
        result = np.zeros((num_players, num_arms, 3), dtype=np.float)
        Win, lose, hit_rate for each machine.
    Output:
        selection_prob: shape=(num_players, num_arms)
        Each row = individual selection probabilities.
        Not normalized.
    """
    num_players = result.shape[0]
    num_arms = result.shape[1]
    # Experienced average for each player. 1 if each machine has never been pulled.
    experienced_average = np.zeros((num_players, num_arms))
    if np.any(result[0,:,0]+result[0,:,1]==0):
        experienced_average[0] = np.ones(num_arms)
    else:
        experienced_average[0] = result[0,:,0]/(result[0,:,0]+result[0,:,1])
    if np.any(result[1,:,0]+result[1,:,1]==0):
        experienced_average[1] = np.ones(num_arms)
    else:
        experienced_average[1] = result[1,:,0]/(result[1,:,0]+result[1,:,1])
    #! This part may be wrong.
    # experienced_average = np.divide(result[:,:,0], result[:,:,0]+result[:,:,1], out=np.ones_like(result[:,:,0]), where=(result[:,:,0]+result[:,:,1])!=0)
    beta_list = np.pi * (result[:,:,0]+result[:,:,1]+2) * np.sqrt((result[:,:,0]+result[:,:,1]+3)/(6*(result[:,:,0]+1)*(result[:,:,1]+1)))
    # Optimal beta for each player.
    beta = beta_list[range(num_players), experienced_average.argsort()[:,::-1][:,num_players]]
    preference = experienced_average * beta.reshape(num_players, 1)
    # Avoid numerical explosion.
    preference -= preference.max(axis=1).reshape(-1,1)
    preference = np.exp(preference)
    selection_prob = np.zeros((num_players, num_arms))
    all_permutations = list(itertools.permutations(range(num_arms)))
    for order in all_permutations:
        # Avoid cancellation of significant digits so that tmp_pref.sum() is not equal to 0.
        tmp_pref = preference.copy()
        tmp_pref[:, order[0]] = 0
        #* Only applicable to 2 players.
        p = preference[:, order[0]]/preference.sum(axis=1) * preference[:, order[1]]/tmp_pref.sum(axis=1)
        selection_prob[:, order[0]] += p/num_players
        selection_prob[:, order[1]] += p/num_players
    selection_prob /= selection_prob.sum(axis=1).reshape(-1,1)
    return selection_prob
