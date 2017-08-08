import preonpy
import numpy as np
from copy import deepcopy
import os

from utils.options import Options
from env.preon_env import Preon_env
from agent.ddpg import DDPG
from utils.memory import ReplayMemory
from agent.evaluator import Evaluator
from utils.util import *


def generate_new_goal(args, validation=False):
    if validation:
        desired_poured_vol = np.random.choice([50,100,150,200,250,300,350,400,450])
        desired_spilled_vol = 0.0
    else:
        desired_poured_vol = np.random.randint(0,args.max_volume)
        # We can only spill so much as available liquid remains
        desired_spilled_vol = np.random.randint(0,args.max_volume - desired_poured_vol)

    desired_poured_vol_norm = (desired_poured_vol - args.max_volume/2.0) / (args.max_volume/2.0)
    desired_spilled_vol_norm = (desired_spilled_vol - args.max_volume/2.0) / (args.max_volume/2.0)
    new_goal = [desired_poured_vol_norm, desired_spilled_vol_norm]
    return new_goal

def train(args, agent, env, evaluate, debug=False):
    if args.visualize:
        vis = args.vis
        visdom_step = 0
        vis.line(X=np.array([0]), Y=np.array([0]), win='Avg_Value_Loss', opts=dict(title='Value loss'))
        vis.line(X=np.array([0]), Y=np.array([0]), win='Avg_Policy_Loss', opts=dict(title='Policy loss'))
        vis.line(X=np.array([0]), Y=np.array([0]), win='Collition_Rate', opts=dict(title='Avg. collition rate (%)'))
        vis.line(X=np.array([0]), Y=np.array([0]), win='Good_Transitions', opts=dict(title='Number of succesful transitions'))
        vis.line(X=np.array([0]), Y=np.array([0]), win='Good_Transitions_Action', opts=dict(title='succesful transitions change state'))
        vis.line(X=np.array([0]), Y=np.array([0]), win='Valid_Reward', opts=dict(title='Validation reward'))
        vis.line(X=np.array([0]), Y=np.array([0]), win='Train_Reward', opts=dict(title='Training reward'))
        vis.scatter(X=np.array([[0,0]]), win='Reached_goals', opts=dict(title="Reached goals"))

    logger = args.logger
    agent.is_training = True
    for epoch in range(args.agent_params.epochs):

        for cycle in range(args.agent_params.cycles):
            episodes_cum_rewards = []
            collision_rate = []
            succesful_transitions = []
            succesful_transitions_with_actions = []
            reached_goals = []
            action_hist = []

            for episode in range(args.agent_params.ep_per_cycle):
                goal = generate_new_goal(args.env_params, validation=True)
                episode_memory = ReplayMemory(env.max_steps)  # Size of memory should be length of episode
                episode_reward = 0.
                collisions_per_episode = 0
                succesful_transitions_episode = 0
                succesful_trans_with_action = 0
                observation = None
                done = False
                while not done:
                    if observation is None:
                        observation, _ = deepcopy(env.reset())
                        agent.reset(observation)

                    # Select an action
                    action = agent.select_action(observation,goal)

                    action_hist.append(action)

                    # Execute the action
                    observation2, reward, done, info, collision = env.step(action, goal)
                    observation2 = deepcopy(observation2)

                    # For performance visualization, we compute the % of collisions
                    if collision:
                        collisions_per_episode += 1
                    elif reward == env.goal_reward:
                        succesful_transitions_episode += 1
                        reached_goals.append(goal)
                        if observation[6:] != observation2[6:]:
                            succesful_trans_with_action += 1

                    # Insert into memory replay
                    agent.observe(goal, reward, observation2, done)

                    # Insert into temporal episode memory
                    episode_memory.push(observation, goal, action, observation2, reward, done)

                    episode_reward += reward
                    observation = deepcopy(observation2)

                episodes_cum_rewards.append(episode_reward)
                # End of episode
                if debug: prGreen('#Epoch: {} Cycle:{} Episode:{} Reward:{}'.format(epoch,cycle,episode,episode_reward))

                # NOTE: This needs to be tested. It is a replacement for the goal generation code below
                counter = 0
                for transition in episode_memory.memory:
                    if transition[0][6:] != transition[3][6:]:
                        new_goal = transition[3][6:]
                        new_reward = env.estimate_new_reward(transition[3],new_goal,transition[4])
                        agent.memory.push(transition[0], new_goal, transition[2], transition[3], new_reward, transition[5])

                        if new_reward == env.goal_reward:
                            succesful_transitions_episode += 1
                            reached_goals.append(new_goal)
                            succesful_trans_with_action += 1
                    else:
                        if counter < len(episode_memory.memory)*0.1:
                            counter += 1
                            new_goal = transition[3][6:]
                            new_reward = env.estimate_new_reward(transition[3],new_goal,transition[4])
                            agent.memory.push(transition[0], new_goal, transition[2], transition[3], new_reward, transition[5])

                            if new_reward == env.goal_reward:
                                succesful_transitions_episode += 1
                                reached_goals.append(new_goal)

                '''
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

                        if new_reward == env.goal_reward:
                            succesful_transitions_episode += 1
                            reached_goals.append(new_goal)
                            if current_transition[0][3:5] != current_transition[3][3:5]:
                                succesful_trans_with_action += 1

                    # 6. Get next transition
                    current_transition, mem_is_empty = episode_memory.remove_first()
                '''

                collision_rate.append(collisions_per_episode*100/float(env.max_steps))
                succesful_transitions.append(succesful_transitions_episode)
                succesful_transitions_with_actions.append(succesful_trans_with_action)

            logger.warning("Reporting @ Epoch: " + str(epoch) + " | Cycle: " + str(cycle) + " | Avg. Reward: " + str(np.mean(episodes_cum_rewards)))
            # End of cycle
            # Training network
            v_losses = []
            p_losses = []
            for step in range(args.agent_params.opt_steps):
                value_loss, policy_loss = agent.update_policy()
                v_losses.append(value_loss)
                p_losses.append(policy_loss)

            # Evaluate performance after training
            if evaluate is not None:
                goal = generate_new_goal(args.env_params, validation=True)
                valid_reward = evaluate(env, agent, goal, debug=debug)
                agent.is_training = True

            # Performance visualization
            avg_collition_rate = np.mean(collision_rate)
            avg_succesful_transitions = np.mean(succesful_transitions)
            avg_succesful_transitions_with_action = np.mean(succesful_transitions_with_actions)
            avg_value_loss = np.mean(v_losses)
            avg_policy_loss = np.mean(p_losses)
            reached_goals = np.array(reached_goals)
            action_hist = np.array(action_hist)

            if args.visualize:
                visdom_step += 1
                vis.line(
                        X=np.array([visdom_step]),
                        Y=np.array([avg_value_loss]),
                        win='Avg_Value_Loss',
                        update='append')
                vis.line(
                        X=np.array([visdom_step]),
                        Y=np.array([avg_policy_loss]),
                        win='Avg_Policy_Loss',
                        update='append')
                vis.line(
                        X=np.array([visdom_step]),
                        Y=np.array([avg_collition_rate]),
                        win='Collition_Rate',
                        update='append')
                vis.line(
                        X=np.array([visdom_step]),
                        Y=np.array([avg_succesful_transitions]),
                        win='Good_Transitions',
                        update='append')
                vis.line(
                        X=np.array([visdom_step]),
                        Y=np.array([valid_reward]),
                        win='Valid_Reward',
                        update='append')
                vis.line(
                        X=np.array([visdom_step]),
                        Y=np.array([np.mean(episodes_cum_rewards)]),
                        win='Train_Reward',
                        update='append')
                vis.line(
                        X=np.array([visdom_step]),
                        Y=np.array([avg_succesful_transitions_with_action]),
                        win='Good_Transitions_Action',
                        update='append')
                vis.scatter(
                        X=reached_goals[:,0],
                        Y=reached_goals[:,1],
                        win='Reached_goals',
                        update='append')
                vis.histogram(X=action_hist[:,0],
                        win='action_x',
                        opts=dict(title='Action X', numbins=20))
                vis.histogram(X=action_hist[:,1],
                        win='action_y',
                        opts=dict(title='Action Y', numbins=20))
                vis.histogram(X=action_hist[:,2],
                        win='action_theta',
                        opts=dict(title='Action Theta', numbins=20))

            # Save model
            agent.save_model(args.model_dir)

        # End of epoch

def test(args, agent, env, goal):
    agent.is_training = False
    agent.eval()

    desired_poured_vol_norm = (goal[0] - args.env_params.max_volume/2.0) / (args.env_params.max_volume/2.0)
    desired_spilled_vol_norm = (goal[1] - args.env_params.max_volume/2.0) / (args.env_params.max_volume/2.0)
    goal = [desired_poured_vol_norm, desired_spilled_vol_norm]

    observation = None

    # reset at the start of episode
    observation, _ = deepcopy(env.reset())
    agent.reset(observation)
    episode_reward = 0.

    assert observation is not None

    # start episode
    done = False
    while not done:
        action = agent.select_action(observation, goal)
        observation2, reward, done, info, collision = env.step(action, goal)
        observation2 = deepcopy(observation2)

        agent.observe(goal, reward, observation2, done)

        # update
        episode_reward += reward
        observation = deepcopy(observation2)

    prYellow('Reporting Validation Reward :{}'.format(episode_reward))
    prYellow('Goal: {} | Final state: {}'.format(goal,observation[6:]))

    env.save_scene(args.root_dir +'/test_scenes/test1.prscene')


if __name__ == "__main__":
    opt = Options()
    np.random.seed(opt.seed)

    # Define environment
    env = Preon_env(opt.env_params)
    _ = env.reset()     # Just to initialize some variables

    # Define agent
    agent = DDPG(opt.agent_params)

    evaluate = Evaluator(opt)

    if opt.mode == 1:
        # Train the model
        train(opt, agent, env, evaluate, debug=True)
    elif opt.mode == 2:
        # Test the model
        goal = [0, 200]     # Defines testing goal in milliliters
        test(opt, agent, env, goal)
