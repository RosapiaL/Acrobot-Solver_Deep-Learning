import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

PATH = r"./"
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env_name = "Acrobot-v1"

env = gym.make(env_name, render_mode="rgb_array")




# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)


steps_done = 0


def select_actionRandom():
    return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

print("TESTING RANDOM")
#Variables for tracking results
episode_durations = []
def render_env(img, title):
    plt.figure(1)
    plt.clf()
    plt.imshow(img)
    plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
      display.display(plt.gcf())
      display.clear_output(wait=True)


def plot_durations(show_result=False):
    plt.figure(2)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
            
frames = []
f1=open("testing_random.txt","a")
num_episodes_test=100
for i_episode in range(1,num_episodes_test+1):         
# Initialize the environment and get it's state
    state, info = env.reset()
    episode_rewards=[]
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    while True:
        action = select_actionRandom()
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        episode_rewards.append(reward)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Move to the next state
        state = next_state

        screen = env.render()
        frames.append(screen)

        if done:
            print("Episode {} Timesteps: {}".format(i_episode, len(frames)))
            f1.write(str(len(frames))+"\n")   
            frames=[]         
            break

print('Completed \n')
f1.close()
print("Saved!\n")

# Initialize the environment and get it's state
state, info = env.reset()
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
while True:
    action = select_actionRandom()
    observation, reward, terminated, truncated, _ = env.step(action.item())
    reward = torch.tensor([reward], device=device)
    screen = env.render()
    render_env(screen, "test_without training")
    episode_rewards.append(reward)
    done = terminated or truncated

    if terminated:
        next_state = None
    else:
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

    # Move to the next state
    state = next_state

    frames.append(screen)
    if done:
        print("Game completed after {} timesteps".format(len(frames)))
        break
print('Completed \n')

plt.figure(figsize=(frames[0].shape[1]/100.0, frames[0].shape[0]/100.0), dpi = 50)
patch = plt.imshow(frames[0])
plt.axis('off')