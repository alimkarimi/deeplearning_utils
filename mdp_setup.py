import numpy as np

def mdp_setup(proj_model, num_states):
    
    x_min, x_max = proj_model[:, 0].min(), proj_model[:, 0].max()
    y_min, y_max = proj_model[:, 1].min(), proj_model[:, 1].max() 
    states_x = np.linspace(x_min, x_max, num=int(np.sqrt(num_states)) + 1)
    states_y = np.linspace(y_min, y_max, num=int(np.sqrt(num_states)) + 1)
    actions = ['u', 'd', 'r', 'l']
    return states_x, states_y, actions