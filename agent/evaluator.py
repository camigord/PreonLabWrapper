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

        agent.is_training = False

        for episode in range(self.num_episodes):
            # reset at the start of episode
            observation, _ = deepcopy(env.reset())
            agent.reset(observation)
            episode_reward = 0.

            assert observation is not None

            # start episode
            done = False
            while not done:
                action = agent.select_action_validation(observation, goal)
                observation2, reward, done, info = env.step(action, goal)
                observation2 = deepcopy(observation2)

                agent.observe(goal, reward, observation2, done)

                # update
                episode_reward += reward
                observation = deepcopy(observation2)

            self.logger.warning("Reporting Validation Reward: " + str(episode_reward))
            self.logger.warning("Validation " + str(episode) + ": Goal=" +str(goal) + " | Final state:" + str(observation[3:5]))

            if debug: prYellow('[Evaluate] #Episode{}: episode_reward:{}'.format(episode,episode_reward))
            result.append(episode_reward)

        agent.is_training = True
        return np.mean(result)
