import numpy as np
from Bench.HPO import HPO
import socketio

class HPORes(HPO):
    def __init__(self, task_id, Seed=0, server_ip='http://192.168.3.41:4999'):
        self.server_ip = server_ip
        self.seed = Seed
        self.name = f'RES_{task_id}'
        self.task_id = task_id
        self.task_type = 'Continuous'

        self.Variable_range = [[0, 1], [0, 1], [0,1], [1,5], [0,5]]
        self.input_dim = len(self.Variable_range)
        self.Variable_type = ['float', 'float', 'float', 'int', 'float', 'int']
        self.Variable_name = ['momentum', 'learning_rate', 'kernel_size', 'weight_decay', 'number_block']
        self.bounds = np.array([[-1.0] * self.input_dim, [1.0] * self.input_dim])


        super(HPORes, self).__init__(
            input_dim=self.input_dim,
            bounds=None,
            RX=self.Variable_range,
            Seed=Seed,
        )

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


    def connect(self):
        self.sio.connect(self.server_ip)



    def initialize_hpo(self, task_name, task_id, seed):
        self.received_data = None
        self.sio.emit('initialize_hpo', data={'task_name': task_name, 'task_id': task_id, 'Seed': seed})
        while self.received_data is None:
            continue

    def f(self, configuration):
        if isinstance(configuration, np.ndarray):
            configuration = configuration.tolist()
        else:
            pass
        self.connect()
        self.initialize_hpo('RES', self.task_id, self.seed)
        self.received_data = None
        self.sio.emit('calculate', {'params': configuration})
        while self.received_data is None:
            continue
        self.sio.emit('disconnect_request')
        self.sio.wait()
        return np.array(self.received_data['result'])


class HPOCNN(HPO):
    def __init__(self, task_id, Seed=0, server_ip = 'http://192.168.3.41:4999'):
        self.server_ip = server_ip
        self.seed = Seed
        self.name = f'MLP_{task_id}'
        self.task_id = task_id

        self.task_type = 'Continuous'

        self.Variable_range = [[0,1], [0,1], [0,1], [0,1]]
        self.input_dim = len(self.Variable_range)
        self.Variable_type = ['float', 'float', 'float', 'float']
        self.Variable_name = ['momentum', 'n_neurals', 'learning_rate', 'activate_weights']
        self.bounds = np.array([[-1.0] * self.input_dim, [1.0] * self.input_dim])

        super(HPOCNN, self).__init__(
            input_dim=self.input_dim,
            bounds=None,
            RX=self.Variable_range,
            Seed=Seed,
        )

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
        self.initialize_hpo('MLP', self.task_id, self.seed)
        self.received_data = None
        self.sio.emit('calculate', {'params': configuration})
        while self.received_data is None:
            continue
        self.sio.emit('disconnect_request')
        self.sio.wait()
        return np.array(self.received_data['result'])


class HPOMLP(HPO):
    def __init__(self, task_id, Seed=0, server_ip = 'http://192.168.3.41:4999'):
        self.server_ip = server_ip
        self.seed = Seed
        self.name = f'MLP_{task_id}'
        self.task_id = task_id
        self.input_dim = 4
        self.task_type = 'Continuous'

        self.Variable_range = [[0,1], [0,1], [0,1], [0,1]]
        self.Variable_type = ['float', 'float', 'float', 'float']
        self.Variable_name = ['momentum', 'n_neurals', 'learning_rate', 'activate_weights']
        self.bounds = np.array([[-1.0] * self.input_dim, [1.0] * self.input_dim])
        self.Variable_name = ['alpha', 'batch_size', 'depth', 'learning_rate_ init', 'width']
        input_dim = len(self.Variable_name)

        super(HPOMLP, self).__init__(
            input_dim=input_dim,
            bounds=None,
            RX=self.Variable_range,
            Seed=Seed,
        )

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
        self.initialize_hpo('MLP', self.task_id, self.seed)
        self.received_data = None
        self.sio.emit('calculate', {'params': configuration})
        while self.received_data is None or self.received_data['result'] == 'wait':
            self.sio.emit()
        self.sio.emit('disconnect_request')
        self.sio.wait()

        return np.array(self.received_data['result'])

if __name__ == '__main__':
    mlp = HPORes(task_id=0,Seed=0)
    mlp.f([[0.1, 0.4, 5,0.001,3]])
