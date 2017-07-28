import preonpy
import numpy as np
from copy import deepcopy

from utils.options import Options
from env.preon_env import Preon_env
from agent.ddpg import DDPG
from utils.memory import ReplayMemory
from agent.evaluator import Evaluator
from utils.util import *

def generate_new_goal(args):
    # np.random.uniform(-1.0,1.0)
    desired_poured_vol = np.random.choice([100,150,200,250,300,350,400,450])
    desired_poured_vol_norm = (desired_poured_vol - args.max_volume/2.0) / (args.max_volume/2.0)
    desired_spilled_vol_norm = (0.0 - args.max_volume/2.0) / (args.max_volume/2.0)
    new_goal = [desired_poured_vol_norm, desired_spilled_vol_norm]
    return new_goal

def train(args, agent, env, evaluate, debug=False):

    logger = args.logger
    agent.is_training = True
    for epoch in range(args.agent_params.epochs):

        for cycle in range(args.agent_params.cycles):
            episodes_cum_rewards = []
            for episode in range(args.agent_params.ep_per_cycle):
                goal = generate_new_goal(args.env_params)
                episode_memory = ReplayMemory(100)  # Size of memory should be length of episode
                episode_reward = 0.
                observation = None
                done = False
                while not done:
                    if observation is None:
                        observation, _ = deepcopy(env.reset())
                        agent.reset(observation)

                    # Select an action
                    action = agent.select_action(observation,goal)

                    # Execute the action
                    observation2, reward, done, info = env.step(action, goal)
                    observation2 = deepcopy(observation2)

                    # Insert into memory replay
                    agent.observe(goal, reward, observation2, done)

                    # Insert into temporal episode memory
                    episode_memory.push(observation, goal, action, observation2, reward, done)

                    episode_reward += reward
                    observation = deepcopy(observation2)

                episodes_cum_rewards.append(episode_reward)
                # End of episode
                if debug: prGreen('#Epoch: {} Cycle:{} Episode:{} Reward:{}'.format(epoch,cycle,episode,episode_reward))

                # Sampling new goals for replay
                # 1. Get first transition
                current_transition, mem_is_empty = episode_memory.remove_first()
                while not mem_is_empty:
                    # 2. Sample new goals from the future states in this episode
                    _, _, _, _, next_s_batch, _ = episode_memory.sample(args.agent_params.k_goals)
                    # 3. For each new goal
                    for state in next_s_batch:
                        new_goal = state[3:5]
                        # 4. Did we reache the new goal in this transition? - Pass next_state, new_goal and reward
                        new_reward = env.estimate_new_reward(current_transition[3],new_goal,current_transition[4])

                        # 5. Store new transition
                        agent.memory.push(current_transition[0], new_goal, current_transition[2], current_transition[3], new_reward, current_transition[5])

                    # 6. Get next transition
                    current_transition, mem_is_empty = episode_memory.remove_first()


            logger.warning("Reporting @ Epoch: " + str(epoch) + " | Cycle: " + str(cycle) + " | Avg. Reward: " + str(np.mean(episodes_cum_rewards)))
            # End of cycle
            for step in range(args.agent_params.opt_steps):
                agent.update_policy()

            logger.warning("Training...")
            # Evaluate performance after training
            if evaluate is not None:
                goal = generate_new_goal(args.env_params)
                evaluate(env, agent, goal, debug=debug)

            # Save model
            agent.save_model(args.model_dir)

        # End of epoch

def test(agent, env, goal, evaluate, debug=False):
    agent.is_training = False
    agent.eval()

    evaluate(env, agent, goal, debug=debug)

if __name__ == "__main__":
    opt = Options()
    np.random.seed(opt.seed)

    # Define environment
    env = Preon_env(opt.env_params)

    # Define agent
    agent = DDPG(opt.agent_params)

    evaluate = Evaluator(opt)

    if opt.mode == 1:
        # Train the model
        train(opt, agent, env, evaluate, debug=True)
    elif opt.mode == 2:
        # Test the model
        goal = [200, 0]     # Defines testing goal in milliliters
        test(agent, env, goal, evaluate, debug=False)
