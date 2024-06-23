# AIMA_page_71_model_based_refelx_agent.py

"""
model-based reflex agent(s) in an environment
AIMA page 71
"""


class Agent:
    """model-based reflex agent"""
    def __init__(self):
        self.state = 0 #the current state the agent thinks it is in
        self.action = "the most recent action taken by the agent"
        self.rules = "if-else rules"

    def sensor_model(self, state, percept):
        # return updated state based on:
        self, state, percept
        return state
    
    def update_state(self, transition_model, sensor_model, state, action, percept):
        state = transition_model(state, action)
        state = sensor_model(state, percept)
        return state
    
    @staticmethod
    def rule_match(state, rules):
        # match a rule given the state (i.e. "reflex" behaviour)
        class Rule: action = lambda self : "action"
        rule = Rule()
        return rule
    
    def get_action(self, percept):
        """
        this block of code is given in the book
        as the 'Model-Based-Reflex-Agent'
        """
        state, action = self.state, self.action
        transition_model = self.environment.transition_model
        sensor_model = self.sensor_model
        
        state = self.update_state(transition_model, sensor_model, state, action, percept)
        rule = self.rule_match(state, self.rules)
        action = rule.action  # action is a function ?
        return action

    def get_reading_from_sensor(self, state):
        percept = "(mangled) signal from the sensor given the state the agent is in"
        return percept
    
    def push_state(self, state):
        """
        this method is called by the environment.
        It presents an agent with the current actual state
        and solicits an action from the agent
        """
        percept = self.get_reading_from_sensor(state)
        action = self.get_action(percept)
        return action



class Environment:
    def __init__(self):
        # mapping of {agent : actual state of agent}
        # this environent can hold multiple agents
        self.agents_states = dict()
    
    def add_agent(self, agent, initial_state=0):
        agent.environment = self
        self.agents_states.update({agent: initial_state})
    
    @staticmethod
    def transition_model(state, action):
        """
        this transition model is accessible both by the environment itself
        and by an agent
        """
        # the new state is just one square further
        new_state = state + 1 if action else 0
        return new_state
    
    def step(self, agent):
        """get the agent to make just one step in this environemnt"""
        # get the actual state the agent is currently in
        state = self.agents_states[agent]
        
        # solicit an action from the agent
        action = agent.push_state(state)
        
        # update the actual state of the agent
        self.agents_states[agent] = self.transition_model(state, action)
    
    def run(self, agent, steps=10):
        for _ in range(steps):
            self.step(agent)
        



if __name__ == '__main__':
    environment = Environment()
    
    agent_1 = Agent()
    agent_2 = Agent()
    
    environment.add_agent(agent_1)
    environment.add_agent(agent_2)
    
    environment.step(agent_1)
    environment.run(agent_2)
    
    print(environment.agents_states)
