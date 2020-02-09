from snake_env import SnakeEnv
import numpy as np

import tensorflow as tf


def create_model(weights=None):
    """创建一个隐藏层为100的神经网络"""
    STATE_DIM, ACTION_DIM = 11, 3
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(120, input_dim=STATE_DIM, activation='relu'),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(ACTION_DIM, activation="linear")
    ])
    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))

    if weights:
        model.load_weights(weights)
    return model


def act(state, epsilon=0.1, step=0):
    """预测动作"""
    return np.argmax(model.predict(np.array([state]))[0])


env = SnakeEnv()
model = create_model("weights.hdf5")
for i in range(1000):
    state = env.reset()
    step = 0
    while True:
        env.render()
        model.predict(np.array([state]))

        done = True
        #next_state, reward, done, _ = env.step(0)
        step += 1
        if done:
            print('Game', i + 1, '      Score:', env.score)
            break
env.close()
