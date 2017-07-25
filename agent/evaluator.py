import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

from util import *

class Evaluator(object):

    def __init__(self, args):
        self.num_episodes = args.validate_steps

    def __call__(self, env, agent, goal, debug=False):
        observation = None
        result = []

        for episode in range(self.num_episodes):
            # reset at the start of episode
            observation, _ = deepcopy(env.reset())
            episode_reward = 0.

            assert observation is not None

            # start episode
            done = False
            while not done:
                action = agent.select_action(observation, goal, decay_epsilon=False)
                observation2, reward, done, info = env.step(action, goal)
                observation2 = deepcopy(observation2)

                # update
                episode_reward += reward

            if debug: prYellow('[Evaluate] #Episode{}: episode_reward:{}'.format(episode,episode_reward))
