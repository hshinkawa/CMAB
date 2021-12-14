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
    ave_convs = []
    std_convs = []
    ave_psms = []
    std_psms = []
    ave_ideals = []
    std_ideals = []
    if not avoid_aem:
        ave_aems = []
        std_aems = []
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    dir_for_output = "data/" + current_time + "/"
    seed_sequence_conv = np.random.randint(np.iinfo(np.int32).max, size=num_trials)
    seed_sequence_psm = np.random.randint(np.iinfo(np.int32).max, size=num_trials)
    seed_sequence_ideal = np.random.randint(np.iinfo(np.int32).max, size=num_trials)
    if not avoid_aem:
        seed_sequence_aem = np.random.randint(np.iinfo(np.int32).max, size=num_trials)
    for num_arms in range(num_arms_min, num_arms_max+1): # Number of machines = 3,4,5.
        os.makedirs(dir_for_output+'{}M/conv/'.format(num_arms))
        os.makedirs(dir_for_output+'{}M/psm/'.format(num_arms))
        os.makedirs(dir_for_output+'{}M/ideal/'.format(num_arms))
        if not avoid_aem:
            os.makedirs(dir_for_output+'{}M/aem/'.format(num_arms))
        env = np.array([0.9, 0.7, 0.5, 0.3] + [0.1]*(num_arms-4))[:num_arms] # Reward envirionment
        envs = list(set(itertools.permutations(env))) # All environments.
        conv_results = np.zeros(len(envs))
        psm_results = np.zeros(len(envs))
        ideal_results = np.zeros(len(envs))
        if not avoid_aem:
            aem_results = np.zeros(len(envs))
        for i, env in enumerate(tqdm(envs)):
            env = np.array(env)
            method = 'conv'
            input_state = generate_input(num_arms, method)
            # Average reward for this env.
            conv_result = np.array(Parallel(n_jobs=-1)([delayed(CMAB)(env, num_selections, input_state, method, seed_sequence_conv[tr]) for tr in range(num_trials)])).mean(axis=0)
            np.save(dir_for_output+'{}M/conv/reward_env_{}.npy'.format(num_arms, i), conv_result)
            conv_results[i] = conv_result.mean()
            method = 'psm'
            input_state = generate_input(num_arms, method)
            psm_result = np.array(Parallel(n_jobs=-1)([delayed(CMAB)(env, num_selections, input_state, method, seed_sequence_psm[tr]) for tr in range(num_trials)])).mean(axis=0)
            np.save(dir_for_output+'{}M/psm/reward_env_{}.npy'.format(num_arms, i), psm_result)
            psm_results[i] = psm_result.mean()
            method = 'ideal'
            input_state = generate_input(num_arms, method)
            ideal_result = np.array(Parallel(n_jobs=-1)([delayed(CMAB)(env, num_selections, input_state, method, seed_sequence_ideal[tr]) for tr in range(num_trials)])).mean(axis=0)
            np.save(dir_for_output+'{}M/ideal/reward_env_{}.npy'.format(num_arms, i), ideal_result)
            ideal_results[i] = ideal_result.mean()
            if not avoid_aem:
                method = 'aem'
                input_state = generate_input(num_arms, method)
                aem_result = np.array(Parallel(n_jobs=-1)([delayed(CMAB)(env, num_selections, input_state, method, seed_sequence_aem[tr]) for tr in range(num_trials)])).mean(axis=0)
                np.save(dir_for_output+'{}M/aem/reward_env_{}.npy'.format(num_arms, i), aem_result)
                aem_results[i] = aem_result.mean()
        ave_convs.append(conv_results.mean())
        std_convs.append(conv_results.std())
        ave_psms.append(psm_results.mean())
        std_psms.append(psm_results.std())
        ave_ideals.append(ideal_results.mean())
        std_ideals.append(ideal_results.std())
        if not avoid_aem:
            ave_aems.append(aem_results.mean())
            std_aems.append(aem_results.std())
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

    if avoid_aem:
        x1 = np.arange(num_arms_max-num_arms_min+1) + 1
        x2 = x1 + 0.25
        x3 = x2 + 0.25
        plt.bar(x1, ave_convs, width=0.25, label='Conv.', align="center")
        plt.bar(x2, ave_psms, width=0.25, label='PSM', align="center")
        plt.bar(x3, ave_ideals, width=0.25, label='Ideal', align="center")
    else:
        x1 = np.arange(num_arms_max-num_arms_min+1) + 1
        x2 = x1 + 0.2
        x3 = x2 + 0.2
        x4 = x3 + 0.2
        plt.bar(x1, ave_convs, width=0.2, label='Conv.', align="center")
        plt.bar(x2, ave_aems, width=0.2, label='AEM', align="center")
        plt.bar(x3, ave_psms, width=0.2, label='PSM', align="center")
        plt.bar(x4, ave_ideals, width=0.2, label='Ideal', align="center")
    label_x = np.arange(num_arms_min, num_arms_max+1)
    plt.legend(loc='upper left')
    plt.xlabel('Number of machines $N$')
    plt.ylabel('Average reward')
    if avoid_aem:
        plt.xticks(x2, label_x)
    else:
        plt.xticks(x2+0.1, label_x)
    plt.savefig(dir_for_output+'average_reward.pdf', bbox_inches="tight", pad_inches=0.05)

    plt.figure(figsize=(8, 6), dpi=80)
    if avoid_aem:
        x1 = np.arange(num_arms_max-num_arms_min+1) + 1
        x2 = x1 + 0.25
        x3 = x2 + 0.25
        plt.bar(x1, std_convs, width=0.25, label='Conv.', align="center")
        plt.bar(x2, std_psms, width=0.25, label='PSM', align="center")
        plt.bar(x3, std_ideals, width=0.25, label='Ideal', align="center")
    else:
        x1 = np.arange(num_arms_max-num_arms_min+1) + 1
        x2 = x1 + 0.2
        x3 = x2 + 0.2
        x4 = x3 + 0.2
        plt.bar(x1, std_convs, width=0.2, label='Conv.', align="center")
        plt.bar(x2, std_aems, width=0.2, label='AEM', align="center")
        plt.bar(x3, std_psms, width=0.2, label='PSM', align="center")
        plt.bar(x4, std_ideals, width=0.2, label='Ideal', align="center")
    label_x = np.arange(num_arms_min, num_arms_max+1)
    plt.legend(loc='upper left')
    plt.xlabel('Number of machines $N$')
    plt.ylabel('Standard deviation\nof the reward')
    if avoid_aem:
        plt.xticks(x2, label_x)
    else:
        plt.xticks(x2+0.1, label_x)
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
