import numpy as np
from multi_env import MultiUAVEnv
import matplotlib.pyplot as plt
import time

def evaluate_random_policy(env, episodes=5, steps=100):
    rewards = []
    overflows = []
    fxs = []
    energies = []

    for ep in range(episodes):
        obs = env.reset()
        ep_reward = 0
        for _ in range(steps):
            actions = env.sample_action()
            obs, rew, done, info = env.step(actions)
            ep_reward += sum(rew)
        rewards.append(ep_reward)
        overflows.append(info['overflow'])
        fxs.append(info['FX'])
        energies.append(info['energy'])

    return {
        'reward': np.mean(rewards),
        'overflow': np.mean(overflows),
        'FX': np.mean(fxs),
        'energy': np.mean(energies)
    }

def evaluate_greedy_policy(env, episodes=5, steps=100):
    def greedy_action(agent_pos, node_pos, node_buffer):
        actions = []
        for pos in agent_pos:
            best_j = np.argmax(node_buffer)
            direction = node_pos[best_j] - pos
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            else:
                direction = np.zeros_like(direction)
            actions.append(direction)
        return actions

    rewards = []
    overflows = []
    fxs = []
    energies = []

    for ep in range(episodes):
        obs = env.reset()
        ep_reward = 0
        for _ in range(steps):
            actions = greedy_action(env.agent_pos, env.node_pos, env.node_buffer)
            obs, rew, done, info = env.step(actions)
            ep_reward += sum(rew)
        rewards.append(ep_reward)
        overflows.append(info['overflow'])
        fxs.append(info['FX'])
        energies.append(info['energy'])

    return {
        'reward': np.mean(rewards),
        'overflow': np.mean(overflows),
        'FX': np.mean(fxs),
        'energy': np.mean(energies)
    }

if __name__ == '__main__':
    env = MultiUAVEnv(render=False)
    print("Evaluating Random Policy...")
    random_result = evaluate_random_policy(env)
    print("Random Policy:", random_result)

    print("\nEvaluating Greedy Policy...")
    greedy_result = evaluate_greedy_policy(env)
    print("Greedy Policy:", greedy_result)
