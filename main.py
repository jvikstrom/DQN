import gym
import sys
import replay_buffer
import numpy as np
import tensorflow as tf

ENV = 'CartPole-v0'

EPOCHS       = 200
ACTION_SPACE = 2
STATE_SPACE  = 4 

FRAME_SZ     = 2000
BATCHSZ      = 50


def train(env, actor, rpbuffer):

    for _ in range(EPOCHS):
        s0       = env.reset()
        terminal = False

        while not terminal:
            env.render()

            action = np.random.choice([0, 1])

            s1, r1, terminal, _ = env.step(action)

            rpbuffer.add((s0, action, r1, terminal, s1))
            s0 = s1


    env.close()


def play(env, actor, games=20):
    for i in range(games):
        terminal = False
        s0 = env.reset()


        while not terminal:
            env.render()
            action = np.random.choice([0, 1])
            s0, _, terminal, _ = env.step(action)

    env.close()


if __name__ == "__main__":

    env      = gym.make(ENV)
    actor    = None
    rpbuffer = replay_buffer.ReplayBuffer(FRAME_SZ)
    
    if "-t" in sys.argv:
        train(env, actor, rpbuffer)

    if "-p" in sys.argv:
        play(env, actor)

