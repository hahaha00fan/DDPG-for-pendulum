from DDPG import Actor, Agent
import gym
import numpy as np
import torch


if __name__ == '__main__':
    
    
    actor = Actor(3, 256, 1)
    actor.load_state_dict(torch.load('actor.pt'))

    env = gym.make('Pendulum-v2')
    env.reset()
    env.render()


    s0 = env.reset()
    
    for step in range(10000):
        env.render()
        s0 = torch.tensor(s0, dtype = torch.float).unsqueeze(0)
        a0 = actor.forward(s0).squeeze(0).detach().numpy()
        s1, _, _, _ = env.step(a0)
        s0 = s1