import mujoco
import numpy as np
from environment.cocel_sip.cocel_sip import SIPAsset
from environment.cocel_sip.viewer import Viewer
from copy import deepcopy
from math import sin, cos

class CoCELSIPENV(SIPAsset):
    def __init__(self):
        super(CoCELSIPENV, self).__init__()
        self.state_dim = 5
        self.action_dim = 1
        self.action_max = 40.0
        self.pos_max = 0.75
        self.viewer = None
        self.eqi_idx = [0, 1, 2, 4]
        self.reg_idx = [3]

    def reset(self):
        self.local_step = 1
        self.total_reward = 0
        q = np.zeros(2)
        q[0] = 0.01 * np.random.randn()
        q[1] = np.pi + .01 * np.random.randn()
        qd = .01 * np.random.randn(2)
        self.state = np.concatenate([q, qd])
        self.prev_state = np.concatenate([q, qd])
        obs = self._get_obs()
        if self.viewer is not None:
            self.viewer.graph_reset()
        return obs, {}

    def step(self, action):
        # RL policy controller
        self.prev_state = self.state.copy()
        self.action_tanh = action / self.action_max
        self._do_simulation(action)
        new_obs = self._get_obs()
        reward, terminated = self._get_reward(new_obs, self.action_tanh)
        info = {}
        self.total_reward += reward
        return new_obs, reward, terminated, False, info

    def _get_reward(self, obs, act):
        pos, cos_th, th_dot = obs[0], obs[3], obs[4]
        notdone = np.isfinite(obs).all() and (np.abs(pos) <= self.pos_max)
        notdone = notdone and np.abs(th_dot) < 27.
        r_pos = 0.5 + 0.5 * np.exp(-0.7 * pos ** 2)
        r_act = 0.8 + 0.2 * np.maximum(1 - (act ** 2), 0.0)
        r_angle = 0.5 - 0.5 * cos_th
        r_vel = 0.5 + 0.5 * np.exp(-0.2 * th_dot ** 2)
        reward = r_pos * r_act * r_angle * r_vel
        done = not notdone
        return reward, done

    def _get_obs(self):
        cart_vel = (self.state[0] - self.prev_state[0]) / self.sample_time
        ang_vel = (self.state[1] - self.prev_state[1]) / self.sample_time
        return np.array([self.state[0], cart_vel, sin(self.state[1] + np.pi),
                         cos(self.state[1] + np.pi), ang_vel])
        # return np.array([self.state[0], self.state[2], sin(self.state[1]+np.pi),
        #                  cos(self.state[1]+np.pi), self.state[3]])

    def render(self):
        if self.viewer is None:
            self._viewer_setup()
        if not self.viewer.is_alive:
            self._viewer_reset()
        qpos = np.array([self.state[0], -self.state[1]])
        qvel = np.array([self.state[2], -self.state[3]])
        self.data.qpos = qpos
        self.data.qvel = qvel
        self.data.userdata[0] = deepcopy(self.action_tanh)
        self.data.userdata[1] = deepcopy(self.total_reward)
        mujoco.mj_forward(self.model, self.data)
        self.viewer.render()

    def _viewer_setup(self):
        self.model = mujoco.MjModel.from_xml_path('./environment/mujoco/cocel_sip.xml')
        self.data = mujoco.MjData(self.model)
        self._viewer_reset()

    def _viewer_reset(self):
        self.viewer = Viewer(model=self.model, data=self.data,
                             width=1100, height=600,
                             title='CoCEL_SIP',
                             hide_menus=True)
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] += 0.3
        self.viewer.cam.elevation += 35
        self.viewer.cam.azimuth = 205