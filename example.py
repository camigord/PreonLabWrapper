import tensorflow as tf
import numpy as np

# Import policy class
from policy import Policy


tf.reset_default_graph()
# TensorFlow required
with tf.Session() as sess:
    policy = Policy(sess)       # Create policy - Restore model defined in Options.py as "save_dir"

    # Define goal
    desired_level = 0.5         # Fill 50% of the cup
    desired_spillage = 0        # Spill 0 ml

    # Set goal (In "theory" could be changed at any moment provided the new goal is still reachable)
    policy.set_goal([desired_level, desired_spillage])

    # TODO: Get current state as a list
    # state = [delta_x, delta_y, theta_angle, previous_action_x, previous_action_y, previous_action_theta, fill_level, spillage, filling_rate]
    state = []

    # Get action from policy - Policy class takes care of normalizing all values accordingly
    a = policy.get_output(state)
