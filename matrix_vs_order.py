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


def main(num_trials, num_selections, avoid_aem, num_arms_min, num_arms_max):
    ave_matrix = []
    std_matrix = []
    ave_order = []
    std_order = []
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    dir_for_output = "data/" + current_time + "-matrix_vs_order/"
    seed_sequence_ideal = np.random.randint(np.iinfo(np.int32).max, size=num_trials)
    for num_arms in range(num_arms_min, num_arms_max+1): # Number of machines = 3,4,5.
        os.makedirs(dir_for_output+'{}M/matrix/'.format(num_arms))
        os.makedirs(dir_for_output+'{}M/order/'.format(num_arms))
        env = np.array([0.9, 0.7, 0.5, 0.3] + [0.1]*(num_arms-4))[:num_arms] # Reward envirionment
        envs = list(set(itertools.permutations(env))) # All environments.
        ideal_results_matrix = np.zeros(len(envs))
        ideal_results_order = np.zeros(len(envs))
        for i, env in enumerate(tqdm(envs)):
            env = np.array(env)
            method = 'ideal'
            input_state = generate_input(num_arms, method)
            joint_selection = 'matrix'
            matrix_result = np.array(Parallel(n_jobs=-1)([delayed(CMAB)(env, num_selections, input_state, method, seed_sequence_ideal[tr], joint_selection=joint_selection) for tr in range(num_trials)])).mean(axis=0)
            np.save(dir_for_output+'{}M/matrix/reward_env_{}.npy'.format(num_arms, i), matrix_result)
            ideal_results_matrix[i] = matrix_result.mean()
            joint_selection = 'order'
            order_result = np.array(Parallel(n_jobs=-1)([delayed(CMAB)(env, num_selections, input_state, method, seed_sequence_ideal[tr], joint_selection=joint_selection) for tr in range(num_trials)])).mean(axis=0)
            np.save(dir_for_output+'{}M/order/reward_env_{}.npy'.format(num_arms, i), order_result)
            ideal_results_order[i] = order_result.mean()
        ave_matrix.append(ideal_results_matrix.mean())
        std_matrix.append(ideal_results_matrix.std())
        ave_order.append(ideal_results_order.mean())
        std_order.append(ideal_results_order.std())
    # Draw figures.
    plt.figure(figsize=(8, 6), dpi=80)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams["font.size"] = 36
    plt.rcParams["legend.fontsize"] = 24
    plt.rcParams['xtick.labelsize'] = 24
    plt.rcParams['ytick.labelsize'] = 24
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['axes.linewidth'] = 1.0

    x1 = np.arange(num_arms_max-num_arms_min+1) + 1
    x2 = x1 + 0.3
    plt.bar(x1, ave_matrix, width=0.3, label='Matrix', align="center")
    plt.bar(x2, ave_order, width=0.3, label='Order', align="center")
    label_x = np.arange(num_arms_min, num_arms_max+1)
    plt.legend(loc='upper left')
    plt.xlabel('Number of machines $N$')
    plt.ylabel('Average reward')
    plt.xticks(x1+0.15, label_x)
    plt.savefig(dir_for_output+'average_reward.pdf', bbox_inches="tight", pad_inches=0.05)

    plt.figure(figsize=(8, 6), dpi=80)
    x1 = np.arange(num_arms_max-num_arms_min+1) + 1
    x2 = x1 + 0.3
    plt.bar(x1, std_matrix, width=0.3, label='Matrix', align="center")
    plt.bar(x2, std_order, width=0.3, label='Order', align="center")
    label_x = np.arange(num_arms_min, num_arms_max+1)
    plt.legend(loc='upper left')
    plt.xlabel('Number of machines $N$')
    plt.ylabel('Standard deviation\nof the reward')
    plt.xticks(x1+0.15, label_x)
    plt.savefig(dir_for_output+'std_reward.pdf', bbox_inches="tight", pad_inches=0.05)


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials", default=100, type=int, help='Number of CMAB trials.')
    parser.add_argument("--num_selections", default=1000, type=int, help='Number of selections in a CMAB trial.')
    parser.add_argument("--avoid_aem", default=True, type=boolean_string, help='Avoid AEM for reduction of time.')
    parser.add_argument("--num_arms_min", default=3, type=int, help='Minimum number of machines.')
    parser.add_argument("--num_arms_max", default=4, type=int, help='Maximum number of machines.')
    args = parser.parse_args()
    main(args.num_trials, args.num_selections, args.avoid_aem, args.num_arms_min, args.num_arms_max)
