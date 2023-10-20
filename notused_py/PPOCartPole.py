import gym
import numpy as np
import torch
import torch.optim as optim

# 创建环境
env = gym.make('CartPole-v1')

# 设置随机种子
s = env.reset(seed=42)
np.random.seed(42)
torch.manual_seed(42)

# 定义模型
model = torch.nn.Sequential(
    torch.nn.Linear(4, 1),
    torch.nn.Sigmoid(),
)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 定义策略
def policy(state):
    with torch.no_grad():
        action_prob = model(torch.tensor(state, dtype=torch.float32))
    return 1 if np.random.rand() < action_prob else 0

# 定义优化目标
def compute_loss(observation):
    action_prob = model(torch.tensor(observation, dtype=torch.float32))
    return -torch.log(action_prob) if policy(observation) == 1 else -torch.log(1 - action_prob)

# 训练模型
for episode in range(1000):
    observation = env.reset()
    episode_loss = 0
    for t in range(1, 10000):
        action = policy(observation)
        observation, reward, done, _ = env.step(action)
        loss = compute_loss(observation)
        episode_loss += loss
        if done:
            break
    optimizer.zero_grad()
    episode_loss.backward()
    optimizer.step()
    if episode % 10 == 0:
        print('Episode {}: loss={}'.format(episode, episode_loss.item()))

# 评估模型
total_reward = 0
for _ in range(100):
    observation = env.reset()
    for t in range(1, 10000):
        action = policy(observation)
        observation, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
print('Average reward: {}'.format(total_reward / 100))

# 关闭环境
env.close()
