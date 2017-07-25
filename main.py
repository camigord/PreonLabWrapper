import numpy as np
from copy import deepcopy

from utils.options import Options
from env.preon_env import *
from agent.ddpg import DDPG
from utils.memory import ReplayMemory

def generate_new_goal(args):
    desired_vol = float(np.random.randint(0,args.max_volume + 1))   # Generate random expected volume.
    new_goal = [desired_vol, 0.0]
    return new_goal

def train(args, agent, env, evaluate, debug=False):

    agent.is_training = True
    for epoch in range(args.epochs):

        for cycle in range(args.cycles):
            for episode in range(args.ep_per_cycle):
                episode_memory = ReplayMemory(100)  # Size of memory should be length of episode
                episode_reward = 0.
                observation = None
                done = False
                while not done:
                    goal = generate_new_goal(args)
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

                # End of episode
                if debug: prGreen('#Epoch: {} Cycle:{} Episode:{} Reward:{}'.format(epoch,cycle,episode,episode_reward))

                # Sampling new goals for replay
                current_transition = episode_memory.remove_first()
                while current_transition is not None:
                    _, _, _, _, next_s_batch, _ = episode_memory.sample(args.k_goals)
                    # Get reward based on new goal and current transition
                    for state in next_s_batch:
                        new_goal = state[3:5]
                        new_reward = env.estimate_new_reward(current_transition[0],new_goal,current_transition[4])

                        # Store new transition
                        agent.memory.push(current_transition[0], new_goal, current_transition[2], state, new_reward, current_transition[5])

                    current_transition = episode_memory.remove_first()

            # End of cycle
            for step in range(args.opt_steps):
                agent.update_policy()

            # Evaluate performance after training
            if evaluate is not None:
                goal = generate_new_goal(args)
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
    env = preon_env(opt.env_params)

    # Define agent
    agent = DDPG(opt.agent_params)

    evaluate = Evaluator(opt.agent_params)

    if opt.mode == 1:
        # Train the model
        train(opt.agent_params, agent, env, evaluate, debug=True):
    elif opt.mode == 2:
        # Test the model
        goal = [200, 0]     # Defines testing goal in milliliters
        test(agent, env, goal, evaluate, debug=True)
