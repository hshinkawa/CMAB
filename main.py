import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from algorithms import ModifiedSoftmax
from joint_selection import joint_matrix, random_order


env = np.array([0.9, 0.5, 0.1]) # Reward envirionment
num_selections = 1000 # Number of selection in a CMAB trial
num_trials = 100 # Number of CMAB trials
num_players = 2 # Number of players
num_arms = len(env) # Number of machines.
input_state = np.array([0, 1/6, 1/6, 1/6, 0, 1/6, 1/6, 1/6, 0])
equality = True
rotation = True


def CMAB(env, num_selections, equality, input_state, rotation):
    selection_probs = np.zeros((num_players, num_arms, num_selections), dtype=np.float)
    selections = np.zeros((num_players, num_selections), dtype=np.int)
    rewards = np.zeros((num_players, num_selections), dtype=np.int)
    result = np.zeros((num_players, num_arms, 3), dtype=np.float)
    for t in range(num_selections):
        # Machine selections.
        selection_prob = ModifiedSoftmax(result)
        selection = joint_matrix(input_state, selection_prob)
        # selection = random_order(input_state, selection_prob)
        selection_probs[:,:,t] = selection_prob
        selections[:,t] = selection

        # Reward results.
        #* Only applicable to 2 players.
        #"""
        wl = np.random.rand(num_players) < env[selection]
        if (selection[0] == selection[1]) & wl[0] & wl[1]: # When the players select the same machine and they both hit.
            result[:, selection[0], 0] += 1
            result[:, selection[0], 2] += 0.5
            rewards[:, t] = 0.5
        else:
            for player in range(num_players):
                if wl[player]:
                    result[player, selection[player], 0] += 1
                    result[player, selection[player], 2] += 1
                    rewards[player, t] = 1
                else:
                    result[player, selection[player], 1] += 1
    # return selection_probs, selections, rewards, result # For detailed analysis.
    return rewards.sum(axis=1)


def detail():
    selection_probs_history = np.zeros((num_players, num_arms, num_selections, num_trials))
    selection_history = np.zeros((num_players, num_selections, num_trials))
    reward_history = np.zeros((num_players, num_selections, num_trials))
    result_history = np.zeros((num_players, num_arms, 3, num_trials)) # Win, lose, hit_rate for each machine.
    for trial in tqdm(range(num_trials)):
        selection_probs, selections, rewards, result = CMAB(env, num_selections, equality, input_state, rotation)
        selection_probs_history[:,:,:,trial] = selection_probs
        selection_history[:,:,trial] = selections
        reward_history[:,:,trial] = rewards
        result_history[:,:,:,trial] = result
    average_reward = reward_history.sum(axis=1, keepdims=True).mean(axis=2).squeeze() # size=(num_players, ). Average reward over the trials for each player.
    return average_reward


def main():
    print(np.array(Parallel(n_jobs=-1)([delayed(CMAB)(env, num_selections, equality, input_state, rotation) for _ in range(num_trials)])).mean(axis=0))


if __name__ == "__main__":
    main()