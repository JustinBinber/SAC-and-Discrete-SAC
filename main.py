import gym
import numpy as np
import torch
import pandas as pd
import time as tm
from learning import Agent
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os

env_name = 'Humanoid-v3'

env = gym.make(env_name)
env_eval = gym.make(env_name)
gamma = 0.99
actor_lr = 3e-4
critic_lr = 3e-4
alpha = 0.12
buffer_size = 150000
batch_size = 256
tau = 0.005
hidden_layer = 256
H = -env.action_space.shape[0]
update_every = 50
update_after = 10000
start_steps = 10000
turns = 3
total_step = 6000000
auto = True
render = False
Save_Module = True
Save_Reward = True
now = tm.localtime()
time = tm.strftime("%Y-%m-%d-%H_%M_%S", now)

step_max = env._max_episode_steps
action_max = float(env.action_space.high[0])
state_num = env.observation_space.shape[0]
action_num = env.action_space.shape[0]

random_seed = 1
torch.manual_seed(random_seed)
np.random.seed(random_seed)


def plot(y):
    episode_idx = range(1, int(total_step / 1000 + 1))
    plt.plot(episode_idx, y)
    plt.xlabel("episode")
    plt.ylabel("MA-reward")
    plt.title("Convergence diagram")
    plt.show()


def evaluate_policy(env, model, turns, render):
    scores = 0
    for j in range(turns):
        s, _ = env.reset()
        done, truncated, ep_r = False, False, 0
        while not done and not truncated:
            # Take deterministic actions at test time
            a = model.actor_net.get_action(s, deterministic=True).cpu().detach().numpy()
            s_, r, done, truncated, _ = env.step(a)
            # r = Reward_adapter(r, EnvIdex)
            ep_r += r
            s = s_
            if render:
                env.render()
        # print(ep_r)
        scores += ep_r
    return scores / turns


if __name__ == '__main__':
    rewards = []
    s, _ = env.reset()
    agent = Agent(state_num, action_num, action_max, hidden_layer,
                  gamma, actor_lr, critic_lr, tau, alpha, batch_size, buffer_size, auto, H)

    # env = gym.wrappers.RecordVideo(env, '/runs/Ant-V3_2')

    for step in range(total_step):
        if step <= start_steps:
            a = env.action_space.sample()
        else:
            a = agent.actor_net.get_action(s, deterministic=False).cpu().detach().numpy()  # 疑似有问题

        s_, r, done, truncated, _ = env.step(a)  # 这里有问题，只能以数组传递
        agent.replay_buffer.push(s, a, r, s_, done)
        s = s_
        if done or truncated:
            s, _ = env.reset()

        if len(agent.replay_buffer) >= update_after and step % update_every == 0:
            for i in range(update_every):
                agent.update()

        if step % 1000 == 0:
            reward_sum = evaluate_policy(env_eval, agent, turns, render=False)
            print(f"step:{step},Reward:{reward_sum}，Alpha:{agent.alpha}")
            rewards.append(reward_sum)

        # if MA30_reward >= 70000:
        # render = True

        if Save_Module:
            model_base = f'models/{env_name}-{time}-alpha={alpha}'
            if not os.path.exists(model_base):
                os.makedirs(model_base)
            if reward_sum >= 50:
                torch.save(agent.actor_net.state_dict(),
                           f'models/{env_name}-{time}-alpha={alpha}-auto={auto}-{env_name}_actor_net-epoch{step}_reward_sum{reward_sum}.pt')
                torch.save(agent.critic_net1.state_dict(),
                           f'models/{env_name}-{time}-alpha={alpha}-auto={auto}-{env_name}-critic_net1-epoch{step}.pt')
                torch.save(agent.critic_net2.state_dict(),
                           f'models/{env_name}-{time}-alpha={alpha}-auto={auto}-{env_name}-critic_net2-epoch{step}.pt')
                torch.save(agent.critic_target_net1.state_dict(),
                           f'models/{env_name}-{time}-alpha={alpha}-auto={auto}-{env_name}-critic_target_net1-epoch{step}.pt')
                torch.save(agent.critic_target_net2.state_dict(),
                           f'models/{env_name}-{time}-alpha={alpha}-auto={auto}-{env_name}-critic_target_net2-epoch{step}.pt')
    if Save_Reward:
        tem = pd.DataFrame(data=rewards)
        tem.to_csv(f'C:/Users/Administrator/Desktop/Data/{env_name}-alpha={alpha}-auto={auto}-{time}.csv', encoding='gbk')

    #plot(rewards)
