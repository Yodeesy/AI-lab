# agent_dir/agent.py
class Agent:
    def __init__(self, env):
        self.env = env

    def init_game_setting(self):
        pass

    def make_action(self, observation, test=False):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError