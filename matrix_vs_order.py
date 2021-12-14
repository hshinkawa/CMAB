import argparse
import itertools
import datetime
import os
import numpy as np
np.random.seed(41)
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from algorithms import ModifiedSoftmax
from joint_selection import joint_matrix, random_order
from main import CMAB


def generate_input(num_arms, method='conv'):
    if method == 'conv' or method == 'psm':
        input_state = np.power(np.sin(np.pi/num_arms * (np.arange(num_arms).reshape(-1,1) - np.arange(num_arms))),2) / num_arms**2
    elif method == 'aem':
        pass
    elif method == 'ideal':
        input_state = np.ones((num_arms,num_arms))/(2*num_arms*(num_arms-1))
        np.fill_diagonal(input_state, 0)
    return input_state


def main(num_trials, num_selections, num_arms_min, num_arms_max):
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    dir_for_output = "data/" + current_time + "-matrix_vs_order/"
    for num_arms in range(num_arms_min, num_arms_max+1): # Number of machines = 3,4,5.
        os.makedirs(dir_for_output+'{}M/matrix/'.format(num_arms))
        os.makedirs(dir_for_output+'{}M/order/'.format(num_arms))
        env = np.array([0.9, 0.7, 0.5, 0.3] + [0.1]*(num_arms-4))[:num_arms] # Reward envirionment
        seed_sequence_matrix = np.random.randint(np.iinfo(np.int32).max, size=num_trials)
        seed_sequence_order = np.random.randint(np.iinfo(np.int32).max, size=num_trials)
        method = 'ideal'
        input_state = generate_input(num_arms, method)
        joint_selection = 'matrix'
        matrix_result = np.array(Parallel(n_jobs=-1)([delayed(CMAB)(env, num_selections, input_state, method, seed_sequence_matrix[tr], joint_selection=joint_selection) for tr in range(num_trials)]))
        np.save(dir_for_output+'{}M/matrix/reward.npy'.format(num_arms), matrix_result)
        joint_selection = 'order'
        order_result = np.array(Parallel(n_jobs=-1)([delayed(CMAB)(env, num_selections, input_state, method, seed_sequence_order[tr], joint_selection=joint_selection) for tr in range(num_trials)])).mean(axis=0)
        np.save(dir_for_output+'{}M/order/reward.npy'.format(num_arms), order_result)


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials", default=100, type=int, help='Number of CMAB trials.')
    parser.add_argument("--num_selections", default=1000, type=int, help='Number of selections in a CMAB trial.')
    parser.add_argument("--num_arms_min", default=3, type=int, help='Minimum number of machines.')
    parser.add_argument("--num_arms_max", default=4, type=int, help='Maximum number of machines.')
    args = parser.parse_args()
    main(args.num_trials, args.num_selections, args.avoid_aem, args.num_arms_min, args.num_arms_max)
