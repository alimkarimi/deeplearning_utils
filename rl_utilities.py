import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

mapping = {0 : 'up',
           1 : 'down',
           2: 'right',
           3: 'left'}
set_actions = {0, 1, 2, 3}
actions = [0,1,2,3]


def create_random_policy(grid_size, num_actions):
    """
    Create a random policy for a 2D gridworld.

    Parameters:
    - grid_size: Tuple representing the size of the grid (rows, columns).
    - num_actions: Number of possible actions for each state.

    Returns:
    - policy: 2D array representing the random policy.
    
    mapping = {0 : 'up',
           1 : 'down',
           2: 'right',
           3: 'left'}
    """ 

    policy = np.random.randint(0, num_actions, size=(grid_size, grid_size))
    
    # make sure policy does not have any "out of bounds" actions
    for r in range(policy.shape[0]):
        for c in range(policy.shape[1]):

            if r == 0: # make sure we do not go up from first row
                if policy[r,c] == 0:
                    policy[r,c] = np.random.randint(1,3, 1)
            if r == 7: # make sure we do not go down from last row
                if policy[r,c] == 1:
                    valid_actions = list(set(actions).difference(set([1])))
                    valid_actions = np.array(valid_actions)
                    policy[r,c] = np.random.choice(valid_actions, p=[1/3, 1/3, 1/3])
            if c == 0: # make sure we do not go left from first col
                if policy[r, c] == 3:
                    valid_actions = list(set(actions).difference(set([3])))
                    valid_actions = np.array(valid_actions)
                    policy[r,c] = np.random.choice(valid_actions, p=[1/3, 1/3, 1/3])
            if c == 7: # make sure we do not go right from last col
                if policy[r,c] == 2:
                    valid_actions = list(set(actions).difference(set([2])))
                    valid_actions = np.array(valid_actions)
                    policy[r,c] = np.random.choice(valid_actions, p=[1/3, 1/3, 1/3])
            # take care of corners:
            
            if r == 0 and c == 0: 
                if policy[r,c] == 0 or policy[r,c] == 3:
                    policy[r,c] = np.random.choice([1,2], p= [0.5, 0.5])
            if r == 0 and c == 7:
                if policy[r,c] == 0 or policy[r,c] == 2:
                    policy[r,c] = np.random.choice([1,3], p = [0.5, 0.5])
            if r == 7 and c == 0:
                if policy[r,c] == 1 or policy[r,c] == 3:
                    policy[r,c] = np.random.choice([0, 2], p = [0.5, 0.5])
            if r == 7 and r == 7:
                if policy[r,c] == 1 or policy[r,c] == 2:
                    policy[r,c] = np.random.choice([0, 3], p = [0.5, 0.5])
                    
    return policy

def choose_init_state(grid_size=8):
    init_state_x = np.random.randint(0, grid_size, 1)
    init_state_y = np.random.randint(0, grid_size, 1)
    init_state = [int(init_state_x), int(init_state_y)]
    return init_state

def choose_action(state, policy):
    """
    state is a 2 element list. Find the corresponding action for that state in the policy
    """
    state_x = state[0] - 1
    state_y = state[1] - 1
    return policy[state_x, state_y]

def choose_image_from_ambiguous(filtered_df):
    """
    If we have a state that has multiple image, choose one
    """
    temp_len = filtered_df.shape[0]
    uniform_probability = 1 / temp_len
    p = np.repeat(uniform_probability, temp_len).tolist()    
    choice = np.random.choice(filtered_df['id'].values.tolist(), 1, p) # randomly pick an image that belongs to the state.
    return choice

def get_image_for_state(df, state):

    filtered_df_test = df[df['state'].map(lambda x: x == state)] # use this to filter for a LIST inside a Pdframe
    #print(filtered_df_test)
    #print('num images from init state', filtered_df_test.shape[0])

    if filtered_df_test.shape[0] > 1:
        choice = choose_image_from_ambiguous(filtered_df_test)
    else:
        choice = filtered_df_test['id'].values
    
    if choice.shape[0] == 0:
        #print('no image in state!!!')
        temp_img = None
        
    #print('this is choice!!', choice)
    if choice.shape[0] != 0: # only get image fp if an image is in that state.
        temp_fp = filtered_df_test.loc[choice, 'fp'].values[0]
        temp_img = Image.open(temp_fp)
    
    return choice, filtered_df_test, temp_img
    
    
def get_next_state(action, prev_state):    
    if action == 0: # up
        next_state = [prev_state[0] - 1, prev_state[1]]
    if action == 1: # down
        next_state = [prev_state[0] + 1, prev_state[1]]
    if action == 2: # right
        next_state = [prev_state[0], prev_state[1] + 1]
    if action == 3: # left
        next_state = [prev_state[0], prev_state[1] - 1]
    print('action was', action , 'and next state was', next_state, '!!!!!')

    return next_state


def generate_trajectory(traj_len, policy):
    for i in range(traj_len):

        if i == 0:
            trajectory = [] # init a list to hold trajectory
            # start at random state:
            prev_state = choose_init_state()
            print('trajectory starts with:', prev_state)
            trajectory.append(prev_state)
            
        # get image from state
        choice, filtered_df_test, temp_img = get_image_for_state(df, prev_state)
        trajectory.append(temp_img)

        #now, we that we have a state, choose an action. Our policy should tell us the mapping.
        temp_action = choose_action(prev_state,policy)
        trajectory.append(temp_action)
        #print('trajectory so far:', trajectory)

        # get new state based on the previous (s,a) pair.
        #print('current state:', prev_state)
        #print('current action!!!', temp_action)

        next_state = get_next_state(temp_action, prev_state)
        trajectory.append(next_state) # add next_state to trajectory
        prev_state = next_state
    
    return trajectory


def display_trajectory(trajectory, title : str):
    fig, ax = plt.subplots(1, 5)
    for n, i in enumerate(trajectory):
        if n % 3 == 1: 
            if i != None:
                temp_img = np.uint8(i)
                ax[n // 3].imshow(temp_img)
                ax[n // 3].set_title(title)
            if i == None:
                temp_img = np.ones((255, 255, 3)) * 255
                ax[n // 3].imshow(temp_img)
                ax[n // 3].set_title(title)
    


