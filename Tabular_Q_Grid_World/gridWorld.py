import colorsys
import random
import time
import tkinter as tk

###################
### User Params ###
###################
gw_width = 10
gw_height = 8

# obstacle locations, this will be a list of tuples, for user inputs, the grid is 1 indexed with the origin
# the bottom right
obstacles = [(2, 2), (2, 3), (3, 4), (9, 3), (2, 4), (2, 5)]
# reward function (reward just depends on state moved to)
reward_location = (10, 8)
reward_value = 1

costs = [(4, 2, -1), (10, 7, -2)]  # ,(4,1,-10)]#(x,y,amount)

discount_factor = .985

# start_location = (1,1)
max_episodes = 300

box_size = 110

################### conversions and stuff
width = gw_width * box_size
height = gw_height * box_size


# convert all to zero index coordinate frame that the canvas uses (origin top left with y axis flipped)
costs = [(x[0] - 1, gw_height-x[1], x[2]) for x in costs]
reward_location = (reward_location[0] - 1, gw_height - reward_location[1])
obstacles = [(x[0] - 1, gw_height-x[1]) for x in obstacles]
cost_values = [x[2] for x in costs]

cost_locations = [(x[0], x[1]) for x in costs]

special_squares = obstacles + cost_locations + [reward_location]
actions = [0, 1, 2, 3]  # up,right,down,left #if you run into an obstacle, the episode is over

###################

def max_2dlist_tuples(input_list):
    max_ = input_list[0][0][0]
    for i in range(len(input_list)):
        for j in range(len(input_list[i])):
            if input_list[i][j] is None:
                continue
            if not isinstance(input_list[i][j], list):
                if input_list[i][j] > max_:
                    max_ = input_list[i][j]
                continue
            for k in range(len(input_list[i][j])):
                if input_list[i][j][k] > max_:
                    max_ = input_list[i][j][k]
    return max_

def min_2dlist_tuples(input_list):
    min_ = input_list[0][0][0]
    for i in range(len(input_list)):
        for j in range(len(input_list[i])):
            if input_list[i][j] is None:
                continue
            if not isinstance(input_list[i][j], list):
                if input_list[i][j] < min_:
                    min_ = input_list[i][j]
                continue
            for k in range(len(input_list[i][j])):
                if input_list[i][j][k] < min_:
                    min_ = input_list[i][j][k]
    return min_

class GridWorld:
    def __init__(self):
        self.root = tk.Tk()
        frame = tk.Frame(self.root)
        frame.pack()

        self.gw = tk.Canvas(self.root, width=width+1, height=height+1, borderwidth=0, highlightthickness=0)
        self.gw.pack()

        # #create horizontal lines
        # for i in range(gw_height+1):
        #     self.gw.create_line(0,i*box_size,width,i*box_size)

        # #create vertical lines
        # for i in range(gw_width+1):
        #     self.gw.create_line(i*box_size,0,i*box_size,height)

        # self.gw.create_polygon([0,0,100,0,0,100], fill = 'red3')

        # draw the grid and all the squares
        for i in range(gw_width):  # i loops through x values
            for j in range(gw_height):  # j loops through y values
                # if not (i,j) in special_squares:
                #     self.gw.create_line(i*box_size,j*box_size,(i+1)*box_size,(j+1)*box_size)
                #     self.gw.create_line((i+1)*box_size,j*box_size,i*box_size,(j+1)*box_size)
                if (i, j) in obstacles:
                    self.gw.create_rectangle(i*box_size, j*box_size, (i+1)*box_size, (j+1)*box_size, fill='black')
                # elif (i,j) in cost_locations:
                #     self.gw.create_rectangle(i*box_size,j*box_size,(i+1)*box_size,(j+1)*box_size, fill = 'red')
                #     self.gw.create_text(i*box_size + box_size/2,j*box_size + box_size/2, text = str(costs[cost_locations.index((i,j))][2]))
                # elif (i,j) in [reward_location]:
                #     self.gw.create_rectangle(i*box_size,j*box_size,(i+1)*box_size,(j+1)*box_size, fill = 'green')
                #     self.gw.create_text(i*box_size + box_size/2,j*box_size + box_size/2, text = str(reward_value))

    def get_fill_val(self, max_qval, min_qval, qval):
        max_qval = float(max_qval) + 0.1
        min_qval = float(min_qval)
        white = colorsys.rgb_to_hsv(0, 0, 0)
        green = colorsys.rgb_to_hsv(0, 255, 0)
        # breakpoint()

        total_range = max_qval - min_qval
        ratio = (qval - min_qval) / total_range
        temp = colorsys.hsv_to_rgb(ratio * 0.333, 1.0, 255)
        return '#' + format(int(temp[0]), 'x').zfill(2) + format(int(temp[1]), 'x').zfill(2) + format(int(temp[2]), 'x').zfill(2)


        if qval == 0:  # white fill
            return '#ffffff'
        if qval > 0 :  # do green, interpolate between white and green IN HSV
            white = colorsys.rgb_to_hsv(0, 0, 0)
            green = colorsys.rgb_to_hsv(0, 255, 0)
            temp = [(1 - qval/max_qval)*white[0] + qval/max_qval*green[0], (1 - qval/max_qval)*white[1] + qval/max_qval*green[1], (1 - qval/max_qval)*white[2] + qval/max_qval*green[2]]
            temp = list(colorsys.hsv_to_rgb(temp[0], temp[1], temp[2]))
            # temp = [(1-qval/max_qval)*255.,(1-qval/max_qval)*255.+qval/max_qval*255,(1-qval/max_qval)*255.]
            return

        temp = [(1-qval/min_qval)*255.+qval/min_qval*255, (1-qval/min_qval)*255., (1-qval/min_qval)*255.]
        return '#' + format(int(temp[0]), 'x').zfill(2) + format(int(temp[1]), 'x').zfill(2) + format(int(temp[2]), 'x').zfill(2)

    def draw_agent(self, coord):
        r = 20
        # x = coord[0] -1
        # y = gw_height-coord[1]
        x = coord[0]
        y = coord[1]
        self.gw.create_oval((x+1/2.)*box_size-r, (y+1/2.)*box_size-r, (x+1/2.)*box_size+r, (y+1/2.)*box_size+r, fill='blue')
        self.root.update_idletasks()
        self.root.update()

    def update_Qvals(self, Qvals):
        max_qval = max_2dlist_tuples(Qvals)
        min_qval = min_2dlist_tuples(Qvals)
        # print(max_qval, min_qval)
        for i in range(gw_width):  # i loops through x values
            for j in range(gw_height):  # j loops through y values
                if not (i, j) in special_squares:
                    # add all the triangles
                    self.gw.create_polygon([i*box_size, j*box_size, (i+1)*box_size, j*box_size, (i+1/2.)*box_size, (j+1/2.)*box_size], fill=self.get_fill_val(max_qval, min_qval, Qvals[i][j][0]),
                                           outline='black', width=2)
                    self.gw.create_polygon([(i+1)*box_size, j*box_size, (i+1)*box_size, (j+1)*box_size, (i+1/2.)*box_size, (j+1/2.)*box_size], fill=self.get_fill_val(max_qval, min_qval, Qvals[i][j][1]),
                                           outline='black', width=2)
                    self.gw.create_polygon([(i+1)*box_size, (j+1)*box_size, i*box_size, (j+1)*box_size, (i+1/2.)*box_size, (j+1/2.)*box_size], fill=self.get_fill_val(max_qval, min_qval, Qvals[i][j][2]),
                                           outline='black', width=2)
                    self.gw.create_polygon([i*box_size, (j+1)*box_size, i*box_size, j*box_size, (i+1/2.)*box_size, (j+1/2.)*box_size], fill=self.get_fill_val(max_qval, min_qval, Qvals[i][j][3]),
                                           outline='black', width=2)

                    # add all the q-value text
                    self.gw.create_text(i*box_size + box_size/2, j*box_size+box_size/5, text=str(round(Qvals[i][j][0], 2)))
                    self.gw.create_text(i*box_size + box_size*4/5, j*box_size+box_size/2, text=str(round(Qvals[i][j][1], 2)))
                    self.gw.create_text(i*box_size + box_size/2, j*box_size+box_size*4/5, text=str(round(Qvals[i][j][2], 2)))
                    self.gw.create_text(i*box_size + box_size/5, j*box_size+box_size/2, text=str(round(Qvals[i][j][3], 2)))
                elif not (i, j) in obstacles:
                    self.gw.create_rectangle(i*box_size, j*box_size, (i+1)*box_size, (j+1)*box_size, fill=self.get_fill_val(max_qval, min_qval, Qvals[i][j]), outline='black', width=2)
                    self.gw.create_text((i+1/2.)*box_size, (j+1/2.)*box_size, text=str(Qvals[i][j]))

        self.root.update_idletasks()
        self.root.update()


gridworld = GridWorld()

# generate the state space and Q-values. The states will be in canvas coordinates.
states = []
Qvals = [[None for i in range(gw_height)] for j in range(gw_width)]
for i in range(gw_width):  # i loops through x values
    for j in range(gw_height):  # j loops through y values
        if not (i, j) in special_squares:
            states.append((i, j))
            Qvals[i][j] = [0, 0, 0, 0]
        elif not (i, j) in obstacles:
            states.append((i, j))
            Qvals[i][j] = 0

num_Qval_updates = list(Qvals)  # make a copy of Qvals, just for storing the number of samples taken in order schedule the learning rate alpha

def dynamics(state, action):  # this is the deterministic transition function
    state_val = states[state]
    if state in (cost_locations + [reward_location]):
        return "terminal"

    if action == 0:
        new_state_val = (state_val[0], state_val[1]-1)
    elif action == 1:
        new_state_val = (state_val[0]+1, state_val[1])
    elif action == 2:
        new_state_val = (state_val[0], state_val[1] + 1)
    elif action == 3:
        new_state_val = (state_val[0]-1, state_val[1])
    else:
        return "invalid action"

    if new_state_val in states:
        return states.index(new_state_val)
    else:
        return state


def get_initial_state():  # later can add the epsilon-greedy stuff to this
    return states.index((0, gw_height-1))
    not_allowed = [reward_location] + cost_locations
    state = random.choice(range(len(states)))  # remember, state is an index
    if states[state] in not_allowed:
        state = get_initial_state()
    return state

def choose_action(state, num_episodes, max_episodes, Qvals):  # epsilon is the probability that you act randomly
    # lets just linearly interpolate to get epsilon going from 100 to 5 percent random
    epsilon = num_episodes/float(max_episodes)*0.05 + (1-num_episodes/float(max_episodes))*1.0
    # epsilon = 0.05
    if random.random() < epsilon:  # act randomly
        return random.choice(actions)
    else:  # pick the action with the higest Qvalue
        return Qvals[states[state][0]][states[state][1]].index(max(Qvals[states[state][0]][states[state][1]]))


# print(Qvals)
max_moves = 1000
num_episodes = 0

while num_episodes <= max_episodes:  # loop through episodes
    num_moves = 0
    # make it such that the initial state cannot be one of the terminal states
    state = get_initial_state()  # initialize with a random initial state. The state is an index into the list of states.


    while num_moves <= max_moves:  # loop through movements and updates
        action = choose_action(state, num_episodes, max_episodes, Qvals)  # sample a random action

        new_state = dynamics(state, action)  # find new state

        # the only terminal states are the ones with rewards (or costs) associated with them
        # eventually combine the rewards and costs into one rewards list formatted like costs
        if states[new_state] == reward_location:
            target = reward_value
            num_moves = max_moves  # hack to move to the next episode
        elif states[new_state] in cost_locations:
            idx = cost_locations.index(states[new_state])
            target = cost_values[idx]
            num_moves = max_moves
        else:  # the case where the new_state is not terminal
            target = 0 + discount_factor * max(Qvals[states[new_state][0]][states[new_state][1]])  # no reward + discount times the max Qvals

        # print(target)

        # update the Qvals
        if num_Qval_updates[states[state][0]][states[state][1]][action] == 0:
            Qvals[states[state][0]][states[state][1]][action] = target
        else:
            alpha = 1./num_Qval_updates[states[state][0]][states[state][1]][action]
            Qvals[states[state][0]][states[state][1]][action] = (1-alpha)*Qvals[states[state][0]][states[state][1]][action] + alpha*target

        state = new_state
        num_moves += 1
    if num_episodes % 1 == 0:
        gridworld.draw_agent(states[new_state])  # change to be in canvas coordinates

        gridworld.update_Qvals(Qvals)
    num_episodes += 1
    if num_episodes % 10 == 0:
        print(f"Finished episode: {num_episodes}")

input('Complete \n press enter to exit')
