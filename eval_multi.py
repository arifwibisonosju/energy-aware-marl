import torch
import numpy as np
from multi_agent_ddpg import DDPGAgent
from multi_env import MultiUAVEnv
import time
import argparse
import matplotlib.pyplot as plt

# === Argument Parser ===
parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, default=100, help='Number of evaluation episodes')
parser.add_argument('--render', action='store_true', help='Render the GUI during evaluation')
parser.add_argument('--model_path', type=str, default=None, help='Optional model path to load weights')
args = parser.parse_args()

# === Parameters ===
N_AGENTS = 3
OBS_DIM = 6
ACT_DIM = 2
MAX_STEPS = 100

# === Initialize Environment and Agents ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = MultiUAVEnv(render=args.render)
agents = [
    DDPGAgent(i, OBS_DIM, ACT_DIM, N_AGENTS, {
        'actor_lr': 1e-3,
        'critic_lr': 1e-3,
        'gamma': 0.95,
        'tau': 0.01,
        'device': device
    }) for i in range(N_AGENTS)
]

# === Optionally Load Models ===
if args.model_path:
    for i, agent in enumerate(agents):
        actor_path = f"{args.model_path}/agent{i}_actor.pth"
        agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
        agent.actor.eval()

# === Logging & Tracking ===
total_rewards = []
total_overflow = []
total_fx = []
total_energy = []

reward_log = open("eval_results.txt", "w")
reward_log.write("Episode,Agent1,Agent2,Agent3,Overflow,FX,Energy\n")

# === Evaluation ===
for ep in range(args.episodes):
    obs_n = env.reset()
    episode_reward = np.zeros(N_AGENTS)
    episode_fx = 0
    episode_overflow = 0
    episode_energy = 0

    for step in range(MAX_STEPS):
        act_n = [agents[i].select_action(obs_n[i], noise=0.0) for i in range(N_AGENTS)]
        next_obs_n, rew_n, done_n, info = env.step(act_n)
        obs_n = next_obs_n
        episode_reward += np.array(rew_n)

        if 'FX' in info:
            episode_fx += info['FX']
        if 'overflow' in info:
            episode_overflow += info['overflow']
        if 'energy' in info:
            episode_energy += info['energy']

        if args.render:
            time.sleep(0.05)

    print(f"[Episode {ep+1}] Rewards: {episode_reward}, Overflow: {episode_overflow}, FX: {episode_fx}, Energy: {episode_energy:.2f}")
    reward_log.write(f"{ep+1},{episode_reward[0]},{episode_reward[1]},{episode_reward[2]},{episode_overflow},{episode_fx},{episode_energy:.2f}\n")

    total_rewards.append(episode_reward)
    total_overflow.append(episode_overflow)
    total_fx.append(episode_fx)
    total_energy.append(episode_energy)

reward_log.close()

# === Summary ===
avg_reward = np.mean(total_rewards, axis=0)
avg_overflow = np.mean(total_overflow)
avg_fx = np.mean(total_fx)
avg_energy = np.mean(total_energy)

print("\n=== Evaluation Summary ===")
print(f"Average Reward per Agent: {avg_reward}")
print(f"Average Overflow per Episode: {avg_overflow:.2f}")
print(f"Average FX per Episode: {avg_fx:.2f}")
print(f"Average Energy per Episode: {avg_energy:.2f}")

# === Enhanced Multi-panel Plot ===
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}
font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}

x = list(range(1, args.episodes + 1))
reward_arr = np.array(total_rewards)

p1 = plt.figure(figsize=(28, 14))

ax1 = p1.add_subplot(2, 4, 1)
ax1.tick_params(labelsize=12)
ax1.grid(linestyle='-.')
for i in range(N_AGENTS):
    ax1.plot(x, reward_arr[:, i], label=f"Agent {i+1}", linewidth=2)
ax1.set_xlabel('Number of evaluation episodes', font1)
ax1.set_ylabel('Reward', font1)
[label.set_fontname('Times New Roman') for label in ax1.get_xticklabels() + ax1.get_yticklabels()]
ax1.legend(prop=font2)

ax2 = p1.add_subplot(2, 4, 2)
ax2.tick_params(labelsize=12)
ax2.grid(linestyle='-.')
ax2.plot(x, total_overflow, marker='o', linewidth=2)
ax2.set_xlabel('Number of evaluation episodes', font1)
ax2.set_ylabel('Overflow Count', font1)
[label.set_fontname('Times New Roman') for label in ax2.get_xticklabels() + ax2.get_yticklabels()]

ax3 = p1.add_subplot(2, 4, 3)
ax3.tick_params(labelsize=12)
ax3.grid(linestyle='-.')
ax3.plot(x, total_fx, marker='s', linewidth=2)
ax3.set_xlabel('Number of evaluation episodes', font1)
ax3.set_ylabel('FX (Out-of-Bounds)', font1)
[label.set_fontname('Times New Roman') for label in ax3.get_xticklabels() + ax3.get_yticklabels()]

ax4 = p1.add_subplot(2, 4, 4)
ax4.tick_params(labelsize=12)
ax4.grid(linestyle='-.')
ax4.plot(x, total_energy, marker='^', linewidth=2)
ax4.set_xlabel('Number of evaluation episodes', font1)
ax4.set_ylabel('Total Energy', font1)
[label.set_fontname('Times New Roman') for label in ax4.get_xticklabels() + ax4.get_yticklabels()]

# Placeholder empty plots for visual balance
for i in range(5, 9):
    ax = p1.add_subplot(2, 4, i)
    ax.axis('off')

plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.savefig("eval_summary_plot.jpg")
plt.show()
