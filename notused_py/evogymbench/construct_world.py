import json
import os
def construct_gym_world(json_name,peak_num, peak_location):
    with open('world_data/simple_walker_env.json', 'r') as f:
        l_dic = json.load(f)
        new_dic = l_dic.copy()
        for key in  l_dic.keys():
            pass
    new_dic = {'grid_width':30, 'grid_height':10, 'objects':{}}
    boj_dic = {}
    indices = list(range(30))
    types = [5] * 30
    for p_id, p in enumerate(range(peak_num)):
        peak_location[p_id]





if __name__ == '__main__':
    print(os.getcwd())
    construct_gym_world(json_name='./world_data/test_walker_world.json', peak_num=1, peak_location=5)