import os
import numpy as np

from Bench.HPO import HPO
os.environ['OMP_NUM_THREADS'] = "1"

import socketio



class HPOXGB(HPO):
    def __init__(self, task_id, Seed=0, server_ip = 'http://192.168.3.45:4999'):
        self.server_ip = server_ip
        self.seed = Seed
        self.name = f'XGB10d_{task_id}'
        self.task_id = task_id
        self.task_type = 'Continuous'
        self.Variable_name = ['eta', 'max_depth','min_child_weight', 'colsample_bytree', 'colsample_bylevel', 'reg_lambda', 'reg_alpha', 'subsample_per_it', 'n_estimators', 'gamma']
        input_dim = len(self.Variable_name)
        self.Variable_range = [[-10, 0], [1, 15],[0, 7], [0.01, 1.0], [0.01, 1], [-10, 10], [-10, 10], [0.1, 1.0], [1, 50], [0,1.0]]
        self.Variable_type = ['float', 'int','float','float','float','float','float','float', 'int', 'float']
        self.log_flag = [True, False, True, False, False, True, True, False, False, False]

        self.sio = socketio.Client()
        self.received_data = None

        @self.sio.on('hpo_initialized')
        def on_hpo_initialized(data):
            print('HPO Initialized:', data)
            self.received_data = data
        @self.sio.on('calculation_result')
        def on_calculation_result(data):
            print('Calculation Result:', data)
            self.received_data = data

        @self.sio.on('disconnected')
        def on_disconnected(data):
            print('Disconnected:', data)
            self.sio.disconnect()


        super(HPOXGB, self).__init__(
            input_dim=input_dim,
            bounds=None,
            RX=self.Variable_range,
            Seed=Seed,
        )


    def connect(self):
        self.sio.connect(self.server_ip)

    def initialize_hpo(self, task_name, task_id, seed):
        self.received_data = None
        self.sio.emit('initialize_hpo', data={'task_name':task_name,'task_id': task_id, 'Seed': seed})
        while self.received_data is None:
            continue

    def f(self, configuration):
        if isinstance(configuration, np.ndarray):
            configuration = configuration.tolist()
        else:
            pass
        self.connect()
        self.initialize_hpo('XGB', self.task_id, self.seed)
        self.received_data = None
        self.sio.emit('calculate', {'params': configuration})
        while self.received_data is None:
            continue
        self.sio.emit('disconnect_request')
        self.sio.wait()
        return np.array(self.received_data['result'])

if __name__ == '__main__':
    task_lists = [167149, 167152, 126029, 167178, 167177]
    problem = HPOXGB(task_id=167149, Seed=0)
    a = problem.f(np.array([[-0.6,0.2,0.3,-0.4,-0.5,0.99,0.7,0.8,0.9,0.2], [0.1,0.2,-0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.8], [0.3,0.2,-0.4,-0.9,0.5,0.7,-0.7,0.8,0.9,0.1]]))
    problem.disconnect()
    print(a)
