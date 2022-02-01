import numpy as np
import itertools
import copy


def joint_matrix(input_state, selection_prob, rng=None):
    """
    Input:
        input_state: generation probabilities at the source.
        selection_prob: individual selection probabilities for each player.
    Output:
        Joint selection. (2,1)/(0,1)...
    """
    num_players = selection_prob.shape[0]
    num_arms = selection_prob.shape[1]
    if rng is None:
        rng = np.random.default_rng()
    selections = list(itertools.product(range(num_arms), repeat=num_players))
    selection_matrix = selection_prob[range(num_players),selections].prod(1)
    selection_matrix = selection_matrix.reshape(num_arms, num_arms)
    selection_matrix /= selection_matrix.sum()
    joint_matrix = selection_matrix * input_state
    joint_matrix = (joint_matrix/joint_matrix.sum()).reshape(-1,)
    return np.array(selections[rng.choice(len(joint_matrix), p=joint_matrix)], dtype=np.int)


def random_order(input_state, selection_prob, rng=None):
    num_players = selection_prob.shape[0]
    num_arms = selection_prob.shape[1]
    if rng is None:
        rng = np.random.default_rng()
    selections = np.array([-1] * num_players, dtype=np.int)
    players_order = rng.permutation(num_players)
    tmp_probs = copy.deepcopy(selection_prob)
    for player in players_order:
        selection = rng.choice(num_arms, p=tmp_probs[player])
        selections[player] = selection
        tmp_probs[:, selection] = 0
        tmp_probs /= tmp_probs.sum(1).reshape(-1,1)
    return selections


def fill_in(min_a, probs_b, used_arm, used_amount):
    n = len(probs_b)
    ind = np.ones(n, dtype=bool)
    ind[used_arm] = False
    remain = probs_b[ind]
    if min_a <= remain[0]:
        remain[0] -= min_a
        probs_b[ind] = remain
        used_arm = np.append(used_arm, np.where(ind==True)[0][0])
        used_amount = np.append(used_amount,min_a)
        min_a = 0
    else:
        amount = remain[0].copy()
        min_a -= amount
        remain[0] = 0
        probs_b[ind] = remain
        used_arm = np.append(used_arm, np.where(ind==True)[0][0])
        used_amount = np.append(used_amount,amount)
        min_a, probs_b, used_arm, used_amount = fill_in(min_a, probs_b, used_arm, used_amount)
    return min_a, probs_b, used_arm, used_amount


def construct_matrix(ori_probs, joint_matrix, remaining_arms):
    probs = ori_probs[:, remaining_arms]
    n = probs.shape[1]
    arm_sum = probs.sum(0)
    min_arm = np.random.choice(np.where(arm_sum==arm_sum.min())[0])
    max_arm = np.where(arm_sum==arm_sum.max())[0]
    max_arm = max_arm[max_arm!=min_arm]
    max_arm = np.random.choice(max_arm)
    mv = probs[:,max_arm].sum()
    if mv>1:
        delta = (mv-1)/(2*(n-1))
        joint_matrix = np.zeros((n,n), dtype=float)
        joint_matrix[:, max_arm] = probs[0]+delta
        joint_matrix[max_arm, :] = probs[1]+delta
        np.fill_diagonal(joint_matrix, 0)
        return joint_matrix
    else:
        if n==3:
            s = probs[0].sum()
            a1 = probs[0,0]
            a2 = probs[0,1]
            a3 = probs[0,2]
            b1 = probs[1,0]
            b2 = probs[1,1]
            b3 = probs[1,2]
            p12 = max([0,a1-b3,-a3+b2])
            joint_matrix[np.ix_(remaining_arms, remaining_arms)] = np.array([
                [0, p12, -p12+a1],
                [s-p12-a3-b3,0,p12-a1+b3],
                [p12+a3-b2,-p12+b2,0]
            ])
            return joint_matrix
        min_a = probs[0, min_arm]
        probs_b = probs[1, :]
        used_arm = np.array([min_arm, max_arm],dtype=int)
        used_amount = np.array([0.0],dtype=float)
        if min_a <= probs_b[max_arm]:
            probs_b[max_arm] -= min_a
            used_amount = np.append(used_amount, min_a)
        else:
            amount = probs_b[max_arm].copy()
            min_a -= amount
            used_amount = np.append(used_amount, amount)
            probs_b[max_arm] = 0
            _, probs_b, used_arm, used_amount = fill_in(min_a, probs_b, used_arm, used_amount)
        joint_matrix[remaining_arms[min_arm], remaining_arms[used_arm]] = used_amount
        min_b = probs[1, min_arm]
        probs_a = probs[0, :]
        used_arm = np.array([min_arm, max_arm],dtype=int)
        used_amount = np.array([0.0],dtype=float)
        if min_b <= probs_a[max_arm]:
            probs_a[max_arm] -= min_b
            used_amount = np.append(used_amount, min_b)
        else:
            amount = probs_a[max_arm].copy()
            min_b -= amount
            used_amount = np.append(used_amount, amount)
            probs_a[max_arm] = 0
            _, probs_a, used_arm, used_amount = fill_in(min_b, probs_a, used_arm, used_amount)
        joint_matrix[remaining_arms[used_arm], remaining_arms[min_arm]] = used_amount
        probs[:, min_arm] = 0
        ori_probs[:, remaining_arms] = probs
        remaining_arms = np.delete(remaining_arms, min_arm)
        return construct_matrix(ori_probs, joint_matrix, remaining_arms)

def fair_matrix(input_state, selection_prob, rng=None):
    probs = selection_prob.copy()
    num_players = selection_prob.shape[0]
    num_arms = selection_prob.shape[1]
    if rng is None:
        rng = np.random.default_rng()
    joint_matrix = np.zeros((num_arms, num_arms), dtype=float)
    remaining_arms = np.arange(num_arms)
    joint_matrix = construct_matrix(probs, joint_matrix, remaining_arms).reshape(-1)
    # 数値誤差による負を解消．
    joint_matrix = np.where(joint_matrix>=0,joint_matrix, 0)
    # 数値誤差による和が1にならない問題を解消．
    joint_matrix /= joint_matrix.sum()
    selections = list(itertools.product(range(num_arms), repeat=num_players))
    return np.array(selections[rng.choice(len(joint_matrix), p=joint_matrix)], dtype=np.int)
