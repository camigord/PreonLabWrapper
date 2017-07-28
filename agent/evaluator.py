import numpy as np
from utils.util import *
from copy import deepcopy

class Evaluator(object):

    def __init__(self, args):
        self.num_episodes = args.agent_params.validate_steps
        self.logger = args.logger

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

            self.logger.warning("Reporting Validation Reward: " + str(episode_reward))
            self.logger.warning("Validation " + str(episode) + ": Goal=" +str(goal) + " | Final state:" + str(observation2[3:5]))

            if debug: prYellow('[Evaluate] #Episode{}: episode_reward:{}'.format(episode,episode_reward))
            result.append(episode_reward)

        return np.mean(result)
