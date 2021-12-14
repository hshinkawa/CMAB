import numpy as np
import itertools
import copy


def joint_matrix(input_state, selection_prob):
    """
    Input:
        input_state: generation probabilities at the source.
        selection_prob: individual selection probabilities for each player.
    Output:
        Joint selection. (2,1)/(0,1)...
    """
    num_players = selection_prob.shape[0]
    num_arms = selection_prob.shape[1]
    selections = list(itertools.product(range(num_arms), repeat=num_players))
    selection_matrix = selection_prob[range(num_players),selections].prod(1)
    selection_matrix = selection_matrix.reshape(num_arms, num_arms)
    selection_matrix /= selection_matrix.sum()
    joint_matrix = selection_matrix * input_state
    joint_matrix = (joint_matrix/joint_matrix.sum()).reshape(-1,)
    return np.array(selections[np.random.choice(len(joint_matrix), p=joint_matrix)], dtype=np.int)


def random_order(input_state, selection_prob, rng=None):
    num_players = selection_prob.shape[0]
    num_arms = selection_prob.shape[1]
    if rng is None:
        rng = np.random.default_rng()
    selections = np.array([-1] * num_players, dtype=np.int)
    players_order = rng.permutation(num_players)
    tmp_probs = copy.deepcopy(selection_prob)
    for player in players_order:
        selection = np.random.choice(num_arms, p=tmp_probs[player])
        selections[player] = selection
        tmp_probs[:, selection] = 0
        tmp_probs /= tmp_probs.sum(1).reshape(-1,1)
    return selections
