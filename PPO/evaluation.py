import json
import os
import gym
import highway_env
import torch
import json

def initial(config):
    model = None
    for root, dirs, files in os.walk('./data/Intersection_ppo_mid'):
        if 'progress.txt' in files:
            config_path = open(os.path.join(root, 'config.json'))
            data_config = json.load(config_path)
            model_path = data_config['logger_kwargs']['output_dir']
            model = torch.load(os.path.join(model_path, 'pyt_save', 'model.pt'))

    env = gym.make(config.env)
    print(env.observation_space.shape)

    env.configure({
        "simulation_frequency": 10,
        "policy_frequency": 2,
        "screen_width": 2000,
        "screen_height": 600,
        "scaling": 6,
    })

    return model, env

def evaluation(agent, env, config):
    crash_counter = 0
    for epoch in range(config.eval_episodes):
        o = env.reset()
        ep_len = 0
        ep_r = 0
        for t in range(config.steps):
            o = o.flatten()
            # a = env.action_space.sample()
            a, _, _, _ = agent.step(torch.as_tensor(o, dtype=torch.float32))

            if config.render:
                env.render()

            next_o, r, d, info = env.step(int(a))
            ep_len += 1
            ep_r += r

            if info['crashed']:
                crash_counter += 1

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == config.steps
            if d or timeout:
                # print('eplen', ep_len)
                # print(ep_r)
                break

        print("当前进度：", epoch / config.eval_episodes * 100, "%")

    return crash_counter

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="intersectionmid-v0")
    parser.add_argument('--eval_episodes', type=int, default=5000)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--print_freq', type=int, default=100)
    # parser.add_argument('--scenario', type=str, default='Intersection')
    parser.add_argument('--algo', type=str, default='ppo')
    config = parser.parse_args()

    agent, env = initial(config)
    crash_counter = evaluation(agent, env, config)

    print('-'*20)
    print("crash counter:", crash_counter)
    print("crash ratio:", crash_counter / config.eval_episodes)