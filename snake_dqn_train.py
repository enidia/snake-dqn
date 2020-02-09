from collections import deque
import random
from snake_env import SnakeEnv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# weigths = "weights.h5"
weigths = None

class DQN(object):
    def __init__(self):
        self.step = 0
        self.update_freq = 200  # 模型更新频率
        self.replay_size = 2000  # 训练集大小
        self.replay_queue = deque(maxlen=self.replay_size)
        self.model = self.create_model(weigths)
        self.target_model = self.create_model(weigths)

    def create_model(self, weights=None):
        """创建一个隐藏层为100的神经网络"""
        STATE_DIM, ACTION_DIM = 11, 3
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(120, input_dim=STATE_DIM, activation='relu'),
            # tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(120, activation='relu'),
            # tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(120, activation='relu'),
            # tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(ACTION_DIM, activation="softmax")
        ])
        model.compile(loss='mean_squared_error',
                      optimizer=tf.keras.optimizers.Adam(0.001))
        if weights:
            model.load_weights(weights)
        return model

    def act(self, s, epsilon=0.1):
        """预测动作"""
        # 刚开始时，加一点随机成分，产生更多的状态
        if np.random.uniform() < epsilon - self.step * 0.0002:
            return np.random.choice([0, 1, 2])
        return np.argmax(self.model.predict(np.array([s]))[0])

    def save_model(self, file_path='weights.h5'):
        print('model saved')
        self.model.save(file_path)

    def remember(self, state, action, next_s, reward):
        """历史记录，position >= 0.4时给额外的reward，快速收敛"""
        """if next_s[0] >= 0.4:
            reward += 1"""
        if next_s[0] + next_s[1] + next_s[2] > 1:
            reward -= 0.001
        self.replay_queue.append((state, action, next_s, reward))

    def train(self, batch_size=64, lr=1, factor=0.95):
        if len(self.replay_queue) < self.replay_size:
            return
        self.step += 1
        # 每 update_freq 步，将 model 的权重赋值给 target_model
        if self.step % self.update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())

        replay_batch = random.sample(self.replay_queue, batch_size)
        s_batch = np.array([replay[0] for replay in replay_batch])
        next_s_batch = np.array([replay[2] for replay in replay_batch])

        Q = self.model.predict(s_batch)
        Q_next = self.target_model.predict(next_s_batch)

        # 使用公式更新训练集中的Q值
        for i, replay in enumerate(replay_batch):
            _, a, _, reward = replay
            Q[i][a] = (1 - lr) * Q[i][a] + lr * (reward + factor * np.amax(Q_next[i]))

        # 传入网络进行训练
        self.model.fit(s_batch, Q, verbose=0)


env = SnakeEnv()
episodes = 1000 # 训练次数

agent = DQN()
for i in range(episodes):
    state = env.reset()
    while True:
        # env.render(speed=0)
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, next_state, reward)
        agent.train()
        state = next_state
        if done:
            print('Game', i + 1, '      Score:', env.score)
            break
    if (i+1) % 10 == 0:
        agent.save_model()
agent.save_model()

env.close()
