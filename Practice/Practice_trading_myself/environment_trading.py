import random
import math
from collections import OrderedDict


class TrafficLight(object):
    """A traffic light that switches periodically."""

    valid_states = [True, False]  # True = NS open; False = EW open

    def __init__(self, state=None, period=None):
        self.state = state if state is not None else random.choice(self.valid_states)
        self.period = period if period is not None else random.choice([2, 3, 4, 5])
        self.last_updated = 0

    def reset(self):
        self.last_updated = 0

    def update(self, t):
        if t - self.last_updated >= self.period:
            self.state = not self.state  # Assuming state is boolean
            self.last_updated = t


class Environment(object):
    """Environment within which all agents operate."""

    valid_actions = [None, 'forward', 'left', 'right']
    valid_inputs = {'light': TrafficLight.valid_states, 'oncoming': valid_actions, 'left': valid_actions, 'right': valid_actions}
    valid_headings = [(1, 0), (0, -1), (-1, 0), (0, 1)]  # E, N, W, S
    
    def __init__(self, verbose=False, num_dummies=100, grid_size = (8, 6)):
        self.verbose = verbose # If debug output should be given

        # Initialize simulation variables
        self.done = False
        self.t = 0
        self.agent_states = OrderedDict()
        self.step_data = {}
        self.success = None

        # Road network
        self.grid_size = grid_size  # (columns, rows)
        self.bounds = (1, 2, self.grid_size[0], self.grid_size[1] + 1)
        self.block_size = 100
        self.hang = 0.6
        self.intersections = OrderedDict()
        self.roads = []
        for x in range(self.bounds[0], self.bounds[2] + 1):
            for y in range(self.bounds[1], self.bounds[3] + 1):
                self.intersections[(x, y)] = TrafficLight()  # A traffic light at each intersection

        for a in self.intersections:
            for b in self.intersections:
                if a == b:
                    continue
                if (abs(a[0] - b[0]) + abs(a[1] - b[1])) == 1:  # L1 distance = 1
                    self.roads.append((a, b))

        # Add environment boundaries
        for x in range(self.bounds[0], self.bounds[2] + 1):
            self.roads.append(((x, self.bounds[1] - self.hang), (x, self.bounds[1])))
            self.roads.append(((x, self.bounds[3] + self.hang), (x, self.bounds[3])))
        for y in range(self.bounds[1], self.bounds[3] + 1):
            self.roads.append(((self.bounds[0] - self.hang, y), (self.bounds[0], y)))
            self.roads.append(((self.bounds[2] + self.hang, y), (self.bounds[2], y)))    

        # Primary agent and associated parameters
        self.primary_agent = None  # to be set explicitly
        self.enforce_deadline = False

        # Trial data (updated at the end of each trial)
        self.trial_data = {
            'testing': False, # if the trial is for testing a learned policy
            'initial_distance': 0,  # L1 distance from start to destination
            'initial_deadline': 0,  # given deadline (time steps) to start with
            'net_reward': 0.0,  # total reward earned in current trial
            'final_deadline': None,  # deadline value (time remaining) at the end
            'actions': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}, # violations and accidents
            'success': 0  # whether the agent reached the destination in time
        }

    def create_agent(self, agent_class, *args, **kwargs):
        """ When called, create_agent creates an agent in the environment. """

        agent = agent_class(self, *args, **kwargs)
        self.agent_states[agent] = {'location': list(random.choice(self.intersections.keys())), 'heading': (0, 1)}
        return agent

    def set_primary_agent(self, agent, enforce_deadline=False):
        """ When called, set_primary_agent sets 'agent' as the primary agent.
            The primary agent is the smartcab that is followed in the environment. """

        self.primary_agent = agent
        agent.primary_agent = True
        self.enforce_deadline = enforce_deadline


    def act(self, agent, action):
        """ Consider an action and perform the action if it is legal.
            Receive a reward for the agent based on traffic laws. """

        assert agent in self.agent_states, "Unknown agent!"
        assert action in self.valid_actions, "Invalid action!"

        state = self.agent_states[agent]
        location = state['location']
        heading = state['heading']
        light = 'green' if (self.intersections[location].state and heading[1] != 0) or ((not self.intersections[location].state) and heading[0] != 0) else 'red'
        inputs = self.sense(agent)

        # Assess whether the agent can move based on the action chosen.
        # Either the action is okay to perform, or falls under 4 types of violations:
        # 0: Action okay
        # 1: Minor traffic violation
        # 2: Major traffic violation
        # 3: Minor traffic violation causing an accident
        # 4: Major traffic violation causing an accident
        violation = 0

        # Reward scheme
        # First initialize reward uniformly random from [-1, 1]
        reward = 2 * random.random() - 1

        # Create a penalty factor as a function of remaining deadline
        # Scales reward multiplicatively from [0, 1]
        fnc = self.t * 1.0 / (self.t + state['deadline']) if agent.primary_agent else 0.0
        gradient = 10
        
        # No penalty given to an agent that has no enforced deadline
        penalty = 0

        # If the deadline is enforced, give a penalty based on time remaining
        if self.enforce_deadline:
            penalty = (math.pow(gradient, fnc) - 1) / (gradient - 1)

        # Agent wants to drive forward:
        if action == 'forward':
            if light != 'green': # Running red light
                violation = 2 # Major violation
                if inputs['left'] == 'forward' or inputs['right'] == 'forward': # Cross traffic
                    violation = 4 # Accident
        
        # Agent wants to drive left:
        elif action == 'left':
            if light != 'green': # Running a red light
                violation = 2 # Major violation
                if inputs['left'] == 'forward' or inputs['right'] == 'forward': # Cross traffic
                    violation = 4 # Accident
                elif inputs['oncoming'] == 'right': # Oncoming car turning right
                    violation = 4 # Accident
            else: # Green light
                if inputs['oncoming'] == 'right' or inputs['oncoming'] == 'forward': # Incoming traffic
                    violation = 3 # Accident
                else: # Valid move!
                    heading = (heading[1], -heading[0])

        # Agent wants to drive right:
        elif action == 'right':
            if light != 'green' and inputs['left'] == 'forward': # Cross traffic
                violation = 3 # Accident
            else: # Valid move!
                heading = (-heading[1], heading[0])

        # Agent wants to perform no action:
        elif action == None:
            if light == 'green' and inputs['oncoming'] != 'left': # No oncoming traffic
                violation = 1 # Minor violation


        # Did the agent attempt a valid move?
        if violation == 0:
            if action == agent.get_next_waypoint(): # Was it the correct action?
                reward += 2 - penalty # (2, 1)
            elif action == None and light != 'green': # Was the agent stuck at a red light?
                reward += 2 - penalty # (2, 1)
            else: # Valid but incorrect
                reward += 1 - penalty # (1, 0)

            # Move the agent
            if action is not None:
                location = ((location[0] + heading[0] - self.bounds[0]) % (self.bounds[2] - self.bounds[0] + 1) + self.bounds[0],
                            (location[1] + heading[1] - self.bounds[1]) % (self.bounds[3] - self.bounds[1] + 1) + self.bounds[1])  # wrap-around
                state['location'] = location
                state['heading'] = heading
        # Agent attempted invalid move
        else:
            if violation == 1: # Minor violation
                reward += -5
            elif violation == 2: # Major violation
                reward += -10
            elif violation == 3: # Minor accident
                reward += -20
            elif violation == 4: # Major accident
                reward += -40

        # Did agent reach the goal after a valid move?
        if agent is self.primary_agent:
            if state['location'] == state['destination']:
                # Did agent get to destination before deadline?
                if state['deadline'] >= 0:
                    self.trial_data['success'] = 1
                
                # Stop the trial
                self.done = True
                self.success = True

                if(self.verbose == True): # Debugging
                    print ("Environment.act(): Primary agent has reached destination!")

            if(self.verbose == True): # Debugging
                print ("Environment.act() [POST]: location: {}, heading: {}, action: {}, reward: {}".format(location, heading, action, reward))

            # Update metrics
            self.step_data['t'] = self.t
            self.step_data['violation'] = violation
            self.step_data['state'] = agent.get_state()
            self.step_data['deadline'] = state['deadline']
            self.step_data['waypoint'] = agent.get_next_waypoint()
            self.step_data['inputs'] = inputs
            self.step_data['light'] = light
            self.step_data['action'] = action
            self.step_data['reward'] = reward
            
            self.trial_data['final_deadline'] = state['deadline'] - 1
            self.trial_data['net_reward'] += reward
            self.trial_data['actions'][violation] += 1

            if(self.verbose == True): # Debugging
                print ("Environment.act(): Step data: {}".format(self.step_data))
        return reward


class Agent(object):
    """Base class for all agents."""

    def __init__(self, env):
        self.env = env
        self.state = None
        self.next_waypoint = None
        self.primary_agent = False

    def reset(self, destination=None, testing=False):
        pass

    def update(self):
        pass

    def get_state(self):
        return self.state

    def get_next_waypoint(self):
        return self.next_waypoint  
