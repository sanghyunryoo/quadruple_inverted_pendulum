import numpy as np
from environment.cocel_sip_v1 import CoCELSIPENV
from environment.cocel_qip_v1 import CoCELQIPENV

class ENV(object):

    def __init__(self, name, n_history, render_mode="human"):
        self.task_name = name
        if name == "CoCELSIP":
            self.env = CoCELSIPENV()
        elif name == "CoCELQIP":
            self.env = CoCELQIPENV()
        elif name == "QuadrupleInvertedPendulum-v4":
            import gymnasium as gym
            import os
            xml_file = os.getcwd() + "/environment/mujoco/quadruple_inverted_pendulum.xml"
            self.env = gym.make("QuadrupleInvertedPendulum-v4", model_path=xml_file, frame_skip=2, render_mode=render_mode)
        else:
            raise ValueError("[Configuration Error] Choose a valid task name.")
        self.n_history = n_history
        self.state_dim = self.env.state_dim * self.n_history
        self.action_dim = self.env.action_dim
        self.action_max = self.env.action_max
        eqi_idx = np.array([self.env.eqi_idx] * n_history)
        eqi_idx += self.env.state_dim * np.arange(n_history).reshape([-1, 1])
        self.eqi_idx = eqi_idx.reshape([-1]).tolist()
        reg_idx = np.array([self.env.reg_idx] * n_history)
        reg_idx += self.env.state_dim * np.arange(n_history).reshape([-1, 1])
        self.reg_idx = reg_idx.reshape([-1]).tolist()

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        self.state[self.env.state_dim:] = self.state[:-self.env.state_dim]
        self.state[:self.env.state_dim] = next_state
        return self.state.reshape([-1]).astype(np.float32).copy(), reward, terminated, truncated, info

    def reset(self):
        init_obs, _ = self.env.reset()
        self.state = np.concatenate([init_obs]*self.n_history)
        return self.state.reshape([-1]).astype(np.float32).copy(), {}

    def max_step(self):
        return 1000

    def render(self):
        if self.task_name in ["CoCELSIP", "CoCELQIP"]:
            self.env.render()
        else:
            pass