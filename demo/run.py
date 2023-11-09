import logging
import os
import argparse
import pickle
import numpy as np

from Method import TMTGP
# from Method import Incremental
# from Method import WeightedSum
# from Method import MultiTask
# from Method import ELLABO
# from Method import Ablation
# from Method import MetaLearning
# from Method import LFL_TREE
# from Method import Restart
# from Method import HEBO
# from Method import TPE

from Benchmark import ConstructTestSuits

from KnowledgeBase.KnowledgeBase import KnowledgeBase
os.environ['MKL_NUM_THREADS'] = "1"
os.environ['NUMEXPR_NUM_THREADS'] = "1"
os.environ['OMP_NUM_THREADS'] = "1"

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-d", "--dim", type=int, default=10)  # 设置维度
parser.add_argument("-n", "--name", type=str, default='test')  # 实验名称，保存在experiments中
parser.add_argument("-s", "--Seed", type=int, default=1)  # 设置随机种子，与迭代次数相关
parser.add_argument("-m", "--Method", type=str, default='TMTGP')  # 设置method:WS,MT,INC
parser.add_argument("-sm", "--save_mode", type=int, default=1)  # 控制是否保存模型
parser.add_argument("-lm", "--load_mode", type=int, default=0)  # 控制是否从头开始
parser.add_argument("-lg", "--load_gym", type=int, default=1)  #
parser.add_argument("-mt", "--match", type=int, default=1)  #
parser.add_argument("-ac", "--acquisition_func", type=str, default='LCB')  # 控制BO的acquisition function
args = parser.parse_args()


task_list_2d = [
    # 'Ackley_3_s',
    # 'MPB5_3_s',
    # 'Griewank_3_s',
    # 'DixonPrice_3_s',
    # 'Rosenbrock_3_s',
    # 'RotatedHyperEllipsoid_3_s',
    # 'RES_3_s'
    'SVM_3',
    ]

task_list_4d = [
    # 'MLP_3_s',
    'RES_3',
    ]
task_list_6d = [
    'Rosenbrock_3_s'
    ]

task_list_10d = [
    'Ackley_3_10',
    'MPB5_3_10',
    'Griewank_3_10',
    'DixonPrice_3_10',
    'Rosenbrock_3_10',
    'RotatedHyperEllipsoid_3_10',
    'lunar_3_10',
    'XGB_3_10',
]


model_dic = {'LFLT':'Tree','TMTGP':'MOGP', 'INC':'MHGP', 'GYM':'GP',
             'WS':'SGPT_M', 'ELLA':'GP', 'MT':'MOGP', 'BO':'GP',
             'Meta':'HyperBO', 'RF':'RF', 'HEBO':'HEBO', 'TPE':'TPE', 'test_BO':'GP', 'MixMTGP':'MOGP',
             'abl1':'MOGP', 'abl2':'MOGP', 'abl3':'MOGP', 'abl4':'MOGP', 'case':'MOGP'}


if __name__ == '__main__':
    Seed = args.Seed
    Exp_name = args.name
    Load_mode = args.load_mode
    Save_mode = args.save_mode
    Load_gym = args.load_gym
    Acfun = args.acquisition_func
    match_switch = args.match
    Method = args.Method

    ## Need to change
    Xdim = args.dim
    if Xdim == 2:
        task_list = task_list_2d
    elif Xdim == 4:
        task_list = task_list_4d
    elif Xdim == 6:
        task_list = task_list_6d

    elif Xdim == 10:
        task_list = task_list_10d


    Plt = False
    Normalize_method = 'all'
    model_name = model_dic[Method]

    if Method == 'TMTGP' or Method[:3] == 'abl':
        Init_method = 'LFL'
    else:
        Init_method = 'random'
    # Init_method = 'fix'
    # Exper_folder = '/mnt/data/cola/LFL_experiments/{}'.format(Exp_name)
    Exper_folder = '../LFL_experiments/{}'.format(Exp_name)

    if not os.path.exists('{}/figs'.format(Exper_folder)):
        try:
            os.makedirs('{}/figs'.format(Exper_folder))
        except:
            pass
    if not os.path.exists('{}/data'.format(Exper_folder)):
        try:
            os.makedirs('{}/data'.format(Exper_folder))
        except:
            pass
    if not os.path.exists('{}/log'.format(Exper_folder)):
        try:
            os.makedirs('{}/log'.format(Exper_folder))
        except:
            pass
    if not os.path.exists('{}/model'.format(Exper_folder)):
        try:
            os.makedirs('{}/model'.format(Exper_folder))
        except:
            pass

    PID = os.getpid()
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',
        handlers=[
            logging.FileHandler(Exper_folder + '/log/' + str(PID) + '.txt'),
            logging.StreamHandler()]
    )
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    ##Set Initial Fes
    Init_FEs = 4*Xdim
    ##Set Function Evaluation for each task
    Terminate_criterion = 11 * Xdim
    FEs_list = []
    for task in task_list:
        FEs_list.extend([Terminate_criterion] * int(task.split('_')[1]))

    ini_quantile = 0.5
    knowledge_num = 2
    test_suits = ConstructTestSuits.ConstructLFLTestSuits(tasks=task_list, budget_list=FEs_list)

    # recorder = Record.Recorder

    if Load_mode == 1:
        try:
            with open('{}/{}d/{}_{}/{}_KB.txt'.format(Exper_folder, Xdim, Method,model_name, Seed), 'rb') as f:
                KB = pickle.load(f)
        except:
            print("Can not find file, please set 'lm=0'.")
            print("KB Start from scratch!")
            KB = KnowledgeBase()
    else:
        KB = KnowledgeBase()


    if Method == 'INC':
        logging.info('Runing(' + str(Seed) + '):' +
                     '\tMethod=' + Method +
                     '\tSeed=' + str(Seed) +
                     '\tAcf=' + Acfun +
                     '\txdim=' + str(Xdim))

        Incremental.Incremental(Dty=np.float64, Plt=Plt, Init=Init_FEs, Init_method=Init_method, Normalize_method=Normalize_method, Xdim=Xdim,
                                Env=test_suits, Acf=Acfun, Seed=Seed, Method=Method, model_name=model_name, KB=KB, Save_mode=Save_mode,
                                Exper_folder=Exper_folder)

    if Method == 'WS':
        logging.info('Runing(' + str(Seed) + '):' +
                     '\tMethod=' + Method +
                     '\tSeed=' + str(Seed) +
                     '\tAcf=' + Acfun +
                     '\txdim=' + str(Xdim))

        if model_name == 'SGPT_POE':
            Acfun = 'TAF_POE'
        elif model_name == 'SGPT_M':
            Acfun = 'TAF_M'
        else:
            pass

        WeightedSum.WeightedSum(Dty=np.float64, Plt=Plt, Init=Init_FEs, Init_method=Init_method, Normalize_method=Normalize_method, Xdim=Xdim,
                                Env=test_suits, Acf=Acfun, Seed=Seed, Method=Method, model_name=model_name, KB=KB, Save_mode=Save_mode,
                                Exper_folder=Exper_folder)
    if Method == 'MT':
        logging.info('Runing(' + str(Seed) + '):' +
                     '\tMethod=' + Method +
                     '\tSeed=' + str(Seed) +
                     '\tAcf=' + Acfun +
                     '\txdim=' + str(Xdim))

        MultiTask.MultiTask(Dty=np.float64, Plt=Plt, Init=Init_FEs, Init_method=Init_method, Normalize_method=Normalize_method, Xdim=Xdim,
                            Env=test_suits, Acf=Acfun, Seed=Seed, Method=Method, model_name=model_name, KB=KB, Save_mode=Save_mode,
                            Exper_folder=Exper_folder, source_task_num = knowledge_num)

    if Method == 'ELLA':
        logging.info('Runing(' + str(Seed) + '):' +
                     '\tMethod=' + Method +
                     '\tSeed=' + str(Seed) +
                     '\tAcf=' + Acfun +
                     '\txdim=' + str(Xdim))

        ELLABO.ella(Dty=np.float64, Plt=Plt, Init=Init_FEs, Init_method=Init_method, Normalize_method=Normalize_method, Xdim=Xdim,
                    Env=test_suits, Acf=Acfun, Seed=Seed, Method=Method, model_name=model_name, KB=KB, Save_mode=Save_mode,
                    Exper_folder=Exper_folder)

    if Method == 'Meta':
        logging.info('Runing(' + str(Seed) + '):' +
                     '\tMethod=' + Method +
                     '\tSeed=' + str(Seed) +
                     '\tAcf=' + Acfun +
                     '\txdim=' + str(Xdim))

        MetaLearning.MetaBO(Dty=np.float64, Plt=Plt, Init=Init_FEs, Init_method=Init_method, Normalize_method=Normalize_method, Xdim=Xdim,
                            Env=test_suits, Acf=Acfun, Seed=Seed, Method=Method, model_name=model_name, KB=KB, Save_mode=Save_mode,
                            Exper_folder=Exper_folder)
    if Method == 'TMTGP':
        logging.info('Runing(' + str(Seed) + '):' +
                     '\tMethod=' + Method +
                     '\tSeed=' + str(Seed) +
                     '\tAcf=' + Acfun +
                     '\txdim=' + str(Xdim))

        TMTGP.LFL(Dty=np.float64, Plt=Plt, Init=Init_FEs, Init_method=Init_method, Xdim=Xdim,
                  Env=test_suits, Acf=Acfun, Seed=Seed, Method=Method, model_name=model_name, KB=KB, Save_mode=Save_mode,
                  Exper_folder=Exper_folder, knowledge_num = knowledge_num, ini_quantile=ini_quantile)

    if Method == 'LFLT':
        logging.info('Runing(' + str(Seed) + '):' +
                     '\tMethod=' + Method +
                     '\tSeed=' + str(Seed) +
                     '\tAcf=' + Acfun +
                     '\txdim=' + str(Xdim))

        LFL_TREE.LFL_Tree(Dty=np.float64, Plt=Plt, Init=Init_FEs, Init_method=Init_method, Normalize_method=Normalize_method, Xdim=Xdim,
                          Env=test_suits, Acf=Acfun, Seed=Seed, Method=Method, model_name=model_name, KB=KB, Save_mode=Save_mode,
                          Exper_folder=Exper_folder, knowledge_num = knowledge_num, ini_quantile=ini_quantile)

    if Method == 'RF' or Method == 'BO' or Method == 'GYM':
        Restart.Restart(Dty=np.float64, Plt=Plt, Init=Init_FEs, Init_method=Init_method, Normalize_method=Normalize_method,
                        Xdim=Xdim,
                        Env=test_suits, Acf=Acfun, Seed=Seed, Method=Method, model_name=model_name, KB=KB, Save_mode=Save_mode,
                        Exper_folder=Exper_folder)


    if Method == 'HEBO':
        HEBO.HEBO(Dty=np.float64, Plt=Plt, Init=Init_FEs, Init_method=Init_method, Normalize_method=Normalize_method,
                  Xdim=Xdim,
                  Env=test_suits, Acf=Acfun, Seed=Seed, Method=Method, model_name=model_name, KB=KB, Save_mode=Save_mode,
                  Exper_folder=Exper_folder)

    if Method == 'TPE':
        logging.info('Runing(' + str(Seed) + '):' +
                     '\tMethod=' + Method +
                     '\tSeed=' + str(Seed) +
                     '\tAcf=' + Acfun +
                     '\txdim=' + str(Xdim))

        TPE.TPE(Dty=np.float64, Plt=Plt, Init=Init_FEs, Init_method=Init_method, Normalize_method=Normalize_method, Xdim=Xdim,
                Env=test_suits, Acf=Acfun, Seed=Seed, Method=Method, model_name=model_name, KB=KB, Save_mode=Save_mode,
                Exper_folder=Exper_folder, knowledge_num = knowledge_num, ini_quantile=ini_quantile)

    # if Method == 'test_BO':
    #     test_BO.test_BO(Dty=np.float64, Plt=Plt, Init=Init_FEs, Init_method=Init_method, Normalize_method=Normalize_method,
    #             Xdim=Xdim,
    #             Env=env, Acf=Acfun, Seed=Seed, Method=Method, model_name=model_name, KB=KB, Save_mode=Save_mode,
    #             Exper_folder=Exper_folder)
    #
    # if Method == 'MixMTGP':
    #     MixMTGP.Mix(Dty=np.float64, Plt=Plt, Init=Init_FEs, Init_method=Init_method, Normalize_method=Normalize_method, Xdim=Xdim,
    #             Env=env, Acf='LFL_LCB', Seed=Seed, Method=Method, model_name=model_name, KB=KB, Save_mode=Save_mode,
    #             Exper_folder=Exper_folder, knowledge_num = knowledge_num, ini_quantile=ini_quantile)


    if Method == 'abl1':
        logging.info('Runing(' + str(Seed) + '):' +
                     '\tMethod=' + Method +
                     '\tSeed=' + str(Seed) +
                     '\tAcf=' + Acfun +
                     '\txdim=' + str(Xdim))

        Ablation.Aba1(Dty=np.float64, Plt=Plt, Init=Init_FEs, Init_method=Init_method, Normalize_method=Normalize_method, Xdim=Xdim,
                      Env=test_suits, Acf=Acfun, Seed=Seed, Method=Method, model_name=model_name, KB=KB, Save_mode=Save_mode,
                      Exper_folder=Exper_folder, knowledge_num = knowledge_num, ini_quantile=ini_quantile)

    if Method == 'abl2':
        logging.info('Runing(' + str(Seed) + '):' +
                     '\tMethod=' + Method +
                     '\tSeed=' + str(Seed) +
                     '\tAcf=' + Acfun +
                     '\txdim=' + str(Xdim))

        Ablation.Aba2(Dty=np.float64, Plt=Plt, Init=Init_FEs, Init_method=Init_method, Normalize_method=Normalize_method, Xdim=Xdim,
                      Env=test_suits, Acf=Acfun, Seed=Seed, Method=Method, model_name=model_name, KB=KB, Save_mode=Save_mode,
                      Exper_folder=Exper_folder, source_task_num=knowledge_num, ini_quantile=ini_quantile)

    if Method == 'abl3':
        logging.info('Runing(' + str(Seed) + '):' +
                     '\tMethod=' + Method +
                     '\tSeed=' + str(Seed) +
                     '\tAcf=' + Acfun +
                     '\txdim=' + str(Xdim))

        Ablation.Aba3(Dty=np.float64, Plt=Plt, Init=Init_FEs, Init_method=Init_method, Normalize_method=Normalize_method, Xdim=Xdim,
                      Env=test_suits, Acf=Acfun, Seed=Seed, Method=Method, model_name=model_name, KB=KB, Save_mode=Save_mode,
                      Exper_folder=Exper_folder, knowledge_num = knowledge_num)

    if Method == 'abl4':
        logging.info('Runing(' + str(Seed) + '):' +
                     '\tMethod=' + Method +
                     '\tSeed=' + str(Seed) +
                     '\tAcf=' + Acfun +
                     '\txdim=' + str(Xdim))

        Ablation.Aba4(Dty=np.float64, Plt=Plt, Init=Init_FEs, Init_method=Init_method, Normalize_method=Normalize_method, Xdim=Xdim,
                      Env=test_suits, Acf=Acfun, Seed=Seed, Method=Method, model_name=model_name, KB=KB, Save_mode=Save_mode,
                      Exper_folder=Exper_folder, knowledge_num = knowledge_num)

    if Method == 'case':
        logging.info('Runing(' + str(Seed) + '):' +
                     '\tMethod=' + Method +
                     '\tSeed=' + str(Seed) +
                     '\tAcf=' + Acfun +
                     '\txdim=' + str(Xdim))

        Ablation.toy(Dty=np.float64, Plt=Plt, Init=Init_FEs, Init_method=Init_method, Normalize_method=Normalize_method, Xdim=Xdim,
                     Env=test_suits, Acf=Acfun, Seed=Seed, Method=Method, model_name=model_name, KB=KB, Save_mode=Save_mode,
                     Exper_folder=Exper_folder, knowledge_num = knowledge_num)

