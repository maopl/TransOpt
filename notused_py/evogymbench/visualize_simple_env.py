import gym
import numpy as np
import os
from evogym import sample_robot, EvoSim,EvoWorld,EvoViewer
from evogym import get_full_connectivity
# import envs from the envs folder and register them

import torch


if __name__ == '__main__':

    # create a random robot
    # pass
    body, connections = sample_robot((5,5))

    body = np.array([[3., 3., 3., 3., 3.],
           [3., 3., 3., 3., 3.],
           [3., 3., 0., 3., 3.],
           [3., 3., 0., 3., 3.],
           [3., 3., 0., 3., 3.]])



    # make the SimpleWalkingEnv using gym.make and with the robot information
    world = EvoWorld.from_json(os.path.join('world_data', 'simple_walker_env.json'))
    world.add_from_array(name='robot', structure=body, x=3, y=1)
    sim = EvoSim(world)
    sim.reset()
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')

    env = gym.make('SimpleWalkingEnv-v1', body=body)
    env.reset()
    connectivity = get_full_connectivity(body)
    config = {
        'structure_shape': (5, 5),
        'train_iters': 500,
        'save_path': './save_data',
    }
    # save_path = config['save_path']
    # save_path_structure = os.path.join(save_path, f'generation_{0}', 'structure')
    # save_path_controller = os.path.join(save_path, f'generation_{0}', 'controller')
    # os.makedirs(save_path_structure, exist_ok=True)
    # os.makedirs(save_path_controller, exist_ok=True)
    #
    # save_path_generation = os.path.join(config['save_path'], f'generation_{0}')
    # save_path_structure = os.path.join(save_path_generation, 'structure', f'{0}')
    # save_path_controller = os.path.join(save_path_generation, 'controller')
    # fitness = run_ppo(
    #     structure=(body, connectivity),
    #     termination_condition=TerminationCondition(config['train_iters']),
    #     saving_convention=(save_path_controller, 0),
    # )

    save_path_controller = os.path.join(config['save_path'], "generation_" + str(0), "controller", "robot_0_controller" + ".pt")
    actor_critic, obs_rms = \
        torch.load(save_path_controller,
                   map_location='cpu')

    obs = torch.tensor(env.reset(),dtype=torch.float32).resize(1,74)
    env.render('screen')
    recurrent_hidden_states = torch.zeros(1,
                                          actor_critic.recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)

    total_steps = 0
    reward_sum = 0
    while total_steps < 10000:
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states, masks)
            action = np.array(action)
            obs, reward, done, info = env.step(action)
            env.render(verbose=True)
            obs = torch.tensor(obs ,dtype=torch.float32).resize(1,74)

            masks.fill_(0.0 if (done) else 1.0)
            reward_sum += reward
            if done:
                env.reset()
                # reward_sum = float(reward_sum.numpy().flatten()[0])
                # print(f'\ntotal reward: {round(reward_sum, 5)}\n')
                # reward_sum = 0
        total_steps += 1

    env.close()
