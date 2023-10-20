import os

import numpy as np
import pandas as pds
import matplotlib.pyplot as plt


def combine_result(root_path,quantile, Xdim, Seed_lst):
    plt.clf()
    root_path = root_path + f'count_{quantile}/'
    reco_num = {'p_right':[], 'p_false':[], 'n_right':[], 'n_false':[]}

    for seed in Seed_lst:
        # wb = xlrd.open_workbook(root_path+f'reco_num_{Xdim}d_seed{seed}.xls')
        # sh =wb.sheet_by_name('sheet1')
        # print(sh.nrows)
        # pass
        # # os.rename(root_path+f'reco_num_{Xdim}d_seed{seed}.xlsx', root_path+f'reco_num_{Xdim}d_seed{seed}.xls')
        df = pds.read_excel(root_path+f'reco_num_{Xdim}d_seed{seed}.xlsx')
        value = df.iloc[0]
        for k in reco_num.keys():
            reco_num[k].append(value[k])

        # reco_num['p_right'].append value['p_right']
        # reco_num['p_false'] += value['p_false']
        # reco_num['n_right'] += value['n_right']
        # reco_num['n_false'] += value['n_false']
    reco_df = pds.DataFrame(reco_num)
    reco_df.boxplot()
    plt.ylabel('number')
    plt.savefig(root_path+f'{Xdim}D_q_{quantile}.png')



if __name__ == '__main__':
    Xdim_lst = [2,5,8]
    Exper_floder = '../../LFL_experiments/'
    Seed_list = [0,1,2,3,4,5,6,7,8,9]
    quantile = [0.5]
    for q in quantile:
        for Xdim in Xdim_lst:
            combine_result(f'{Exper_floder}', q, Xdim,Seed_list)