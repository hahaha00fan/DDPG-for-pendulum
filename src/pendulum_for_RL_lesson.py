__author__ = 'fanyu_2021E8018782022'
__email__ = 'fanyu21@mails.ucas.ac.cn'



import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path


class my_PendulumEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self):
        self.max_speed = 15 * np.pi
        self.max_voltage = 3.0
        self.dt = 0.005

        # parameters
        self.g = 9.81
        self.m = 0.055
        self.l = 0.042
        self.J = 1.91e-4
        self.b = 3e-6
        self.K = 0.0536
        self.R = 9.5

        self.viewer = None

        # high limit of observation space
        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)

        self.action_space = spaces.Box(low=-self.max_voltage, high=self.max_voltage, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        # action space: ([-3], [3])
        # observation space: ([-1, -1, -15pi], [1, 1, 15pi])


        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u): # u is the action
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        J = self.J
        b = self.b
        K = self.K
        R = self.R
        dt = self.dt

        u = np.clip(u, -self.max_voltage, self.max_voltage)[0]
        self.last_u = u  # for rendering
        costs = 5 * (angle_normalize(th) ** 2) + 0.1 * (thdot ** 2) + 1 * (u ** 2) # reward function

        newthdot = thdot + (1 / J) *(m * l * np.sin(th) - b * thdot - K * K / R *thdot + K / R * u) * dt # kinetic function
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, 0.2)
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1.0, 1.0)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u is not None:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi