from __future__ import print_function
import sys
 
# setting path
sys.path.append('..')

from datetime import datetime
import numpy as np
import gym
import os
import json
import torch
from model import pyTorchModel
from DN1.DN_full_connect import DN1

def run_episode(env, agent, DN, rendering=True, max_timesteps=1000):
    
    episode_reward = 0
    step = 0

    state = env.reset()
    cnt = 0
    last_act = None
    last_state = None
    while True:
        if cnt < 1:
            DN_state = torch.zeros(7056)
            DN_state[0] = 1.0
            DN_action = torch.zeros(3)
            DN.update(DN_state, DN_action)
            last_act = DN_action
            last_state = DN_state
            cnt += 1
        
        # preprocess
        state = state[:-12,6:-6]
        state = np.dot(state[...,0:3], [0.299, 0.587, 0.114])
        state = state/255.0

        # get action
        agent.eval()
        tensor_state = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0)
        #print("STATE: ", tensor_state.shape)
        tensor_action = agent(tensor_state)
        a = tensor_action.detach().numpy()[0]
        #print("ACTION: ", a)

        DN_state = torch.flatten(tensor_state.detach())
        DN_action = torch.flatten(tensor_action.detach())#torch.zeros(3)
        #if tensor_action[1] >= 0.5:
        #print("DN STATE: ", DN_state.shape)
        #print("DN ACT: ", DN_action.shape)
        response = DN.update(last_state, last_act, supervized_z=DN_action)
        last_state = DN_state
        last_act = DN_action
        #print("RESPONSE: ", response[0:10])

        next_state, r, done, info = env.step(a)   
        episode_reward += r       
        state = next_state
        step += 1
        
        if rendering:
            env.render()

        if done or step > max_timesteps: 
            break

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True                      
    
    n_test_episodes = 15                  # number of episodes to test

    # load agent
    agent = pyTorchModel()
    agent.load_state_dict(torch.load("agent.pkl"))
    DN = DN1(7056, 100, 3)

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, DN, rendering=rendering)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    fname = "results/results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')

    """env = gym.make('CarRacing-v0').unwrapped
    env.reset()

    samples = {
        "state": [],
        "next_state": [],
        "reward": [],
        "action": [],
        "terminal" : [],
    }

    episode_rewards = []
    steps = 0
    while True:
        episode_reward = 0
        state = env.reset()
        while True:

            next_state, r, done, info = env.step(a)
            episode_reward += r

            samples["state"].append(state)            # state has shape (96, 96, 3)
            samples["action"].append(np.array(a))     # action has shape (1, 3)
            samples["next_state"].append(next_state)
            samples["reward"].append(r)
            samples["terminal"].append(done)
            
            state = next_state
            steps += 1

            if steps % 1000 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("\nstep {}".format(steps))

            #if args.collect_data and steps % 5000 == 0:
            #    print('... saving data')
            #    store_data(samples, "./data")
            #    save_results(episode_rewards, "./results")

            env.render()
            if done: 
                break
        
        episode_rewards.append(episode_reward)

    env.close()"""