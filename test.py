# Test file
import numpy as np
import preonpy
import os

#from preon_env import preon_env
from preon_env import *

env = preon_env("scene1.prscene")
state, info = env.reset()

# TODO: adjust velocity units to be /per_frame and not /per_second
action0 = (10.0,0.0,50.0)    # vel_x, vel_y, vel_theta
action1 = (10.0,0.0,175.0)
action2 = (0.0,0.0,0.0)
action3 = (0.0,0.0,0.0)
action4 = (0.0,0.0,-225.0)
action5 = (0.0,0.0,0.0)
action6 = (0.0,0.0,0.0)

actions = [action0, action1, action2, action3, action4, action5, action6]
for action in actions:
     print('Time:', env.get_elapsed_time())
     print('Stats', state, info)
     state, R, info = env.step(action)



env.save_scene(os.getcwd()+'/test1.prscene')
print('End')

'''
print(env.predict_collision((0,0,0)))

pos = (3,3)
angle = 345.0
size = (4, 4)
print(get_vertices(pos, angle, size))
'''
