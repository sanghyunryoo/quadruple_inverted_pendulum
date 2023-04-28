from gymnasium.envs.registration import register

register(
    id='QuadrupleInvertedPendulum-v4',
    entry_point="environment.quadruple_inverted_pendulum_v4:QuadrupleInvertedPendulumEnv",
    max_episode_steps=1000
)