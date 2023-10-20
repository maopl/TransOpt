# import envs and necessary gym packages
from gym.envs.registration import register

# register the env using gym's interface
register(
    id = 'SimpleWalkingEnv-v1',
    entry_point = 'envs.simple_env:SimpleWalkerEnvClass',
    max_episode_steps = 500
)

register(
    id = 'TestWalkingEnv-v0',
    entry_point = 'envs.test_env:TestWalkerEnvClass',
    max_episode_steps = 500
)