import numpy as np
import gym


def lunar_lander_simulation(w, is_continuous=False, print_reward=False, seed=1, steps_limit=1000, timeout_reward=100):
    total_reward = 0.
    task_done = False
    steps = 0
    env_name = "LunarLander-v2"
    env = gym.make(env_name)
    s = env.reset()
    while True:
        if steps > steps_limit:
            total_reward -= timeout_reward
            break
        a = heuristic_controller(s, w)

        s, r, done, info = env.step(a)

        total_reward += r
        steps += 1
        if done:
            task_done = True
            break
    if print_reward:
        print(f"Total reward: {total_reward}")
    return total_reward


def heuristic_controller(s, w, is_continuous=False):
    # w is the array of controller parameters of shape (1, 12)
    angle_target = s[0] * w[0] + s[2] * w[1]
    if angle_target > w[2]:
        angle_target = w[2]
    if angle_target < w[-2]:
        angle_target = -w[2]
    hover_target = w[3] * np.abs(s[0])
    angle_todo = (angle_target - s[4]) * w[4] - (s[5]) * w[5]
    hover_todo = (hover_target - s[1]) * w[6] - (s[3]) * w[7]
    if s[6] or s[7]:
        angle_todo = w[8]
        hover_todo = -(s[3]) * w[9]
    if is_continuous:
        a = np.array([hover_todo * 20 - 1, angle_todo * 20])
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > w[10]:
            a = 2
        elif angle_todo < - w[11]:
            a = 3
        elif angle_todo > + w[11]:
            a = 1
    return a


if __name__ == '__main__':
    x_test = np.array(12 * [0.5])
    obj = lunar_lander_simulation(x_test)
    print("objective is: ", obj)
