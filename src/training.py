from DDPG import Agent
import gym
import numpy as np
import torch

if __name__ == '__main__':

    env = gym.make('Pendulum-v2')
    env.reset()
    env.render()

    params = {
        'env': env,
        'gamma': 0.98,
        'actor_lr': 0.001,
        'critic_lr': 0.001,
        'tau': 0.02,
        'capacity': 10000,
        'batch_size': 32
    }

    agent = Agent(**params)

    record = [-99999]

    for episode in range(2000):
        s0 = env.reset()
        episode_reward = 0

        for step in range(500):
            env.render()
            a0 = agent.act(s0)
            s1, r1, done, _ = env.step(a0)
            agent.put(s0, a0, r1, s1)

            episode_reward += r1
            s0 = s1

            agent.learn()
        if episode_reward >= max(record):
            torch.save(agent.actor.state_dict(), 'actor.pt')
        record.append(episode_reward)
        print(episode, ':', episode_reward)
    
    np.savetxt('record.txt', record)
