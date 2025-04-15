
from typing import Dict, List
import numpy as np
import time
import matplotlib.pyplot as plt

from transopt.optimizer.optimizer_base.evo import EVOBase
from transopt.benchmark.Synthetic.MovingPeakBenchmark import MovingPeakGenerator
from transopt.benchmark.problem_base.transfer_problem import TransferProblem, RemoteTransferOptBenchmark
from transopt.space.search_space import SearchSpace

class mQSO(EVOBase):
    def __init__(self, pop_size=5, c1=2.05, c2=2.05, seed=0, config={}):
        super(mQSO, self).__init__(config=config)
        self.seed = seed
        np.random.seed(seed)
        self.pop_size = pop_size
        self.c1 = c1
        self.c2 = c2
        self.swarm_number = config.get('swarm_number', 10)
        self.exclusion_limit = config.get('exclusion_limit', 0.5)
        self.convergence_limit = self.exclusion_limit
        self.quantum_number = config.get('quantum_number', 5)
        self.shift_severity = config.get('shift_severity', 1)
        self.quantum_radius = self.shift_severity 

        self.diversity_plus = config.get('diversity_plus', 1)
        
        self.swarms = None
        self.min_coordinate = None
        self.max_coordinate = None
        self.dimension = None
        
        self.visualization_flag = 0

        self.iteration = 0
        
        self.BestErrorBeforeChange = []
        self.OfflineError = []
        self.CurrentError = []
        self.Runtime = []
        
    
    def link_task(self, task_name:str, search_space: SearchSpace):
        super().link_task(task_name, search_space)
        self.dimension = len(search_space.ranges)
        self.min_coordinate = np.array([search_space.ranges[v][0] for v in search_space.variables_order])
        self.max_coordinate = np.array([search_space.ranges[v][1] for v in search_space.variables_order])



    def initilization(self):
        if self.swarms is None:
            self.swarms = []
            for _ in range(self.swarm_number):
                pop = self.subpopulation_generator()
                self.swarms.append(pop)
        

    def subpopulation_generator(self):
        pop = {}
        pop['Gbest_past_environment'] = np.full((self.dimension,), np.nan)
        pop['Velocity'] = np.zeros((self.pop_size, self.dimension))
        pop['Shifts'] = []
        pop['X'] = self.min_coordinate + (self.max_coordinate - self.min_coordinate) * np.random.rand(self.pop_size, self.dimension)

        pop['PbestValue'] = np.zeros((self.pop_size,))
        pop['PbestPosition'] = np.zeros((self.pop_size, self.dimension))
        pop['BestValue'] = -np.inf
        pop['BestPosition'] = np.zeros((1, self.dimension))

        
        return pop

    def update_swarm(self, swarm_id, swarm, fitness):
        self.swarms[swarm_id]['FitnessValue'] = np.array([i['f1'] for i in fitness])
        for j in range(self.pop_size):
            if self.swarms[swarm_id]['FitnessValue'][j] > self.swarms[swarm_id]['PbestValue'][j]:
                self.swarms[swarm_id]['PbestValue'][j] = self.swarms[swarm_id]['FitnessValue'][j]
                self.swarms[swarm_id]['PbestPosition'][j, :] = self.swarms[swarm_id]['X'][j, :].copy()
        best_index = np.argmax(self.swarms[swarm_id]['PbestValue'])
        best_pbest_value = self.swarms[swarm_id]['PbestValue'][best_index]
        if best_pbest_value > self.swarms[swarm_id]['BestValue']:
            self.swarms[swarm_id]['BestValue'] = best_pbest_value
            self.swarms[swarm_id]['BestPosition'] = self.swarms[swarm_id]['PbestPosition'][best_index].copy()
        
        return


    def reset_swarm(self, swarm_id):
        self.swarms[swarm_id]['FitnessValue'] = -np.inf * np.ones((self.pop_size,))
        self.swarms[swarm_id]['PbestValue'] = self.swarms[swarm_id]['FitnessValue'].copy()
        best_index = np.argmax(self.swarms[swarm_id]['PbestValue'])
        self.swarms[swarm_id]['BestValue'] = self.swarms[swarm_id]['PbestValue'][best_index]
        self.swarms[swarm_id]['BestPosition'] = self.swarms[swarm_id]['PbestPosition'][best_index].copy()
        return
    
    

    def observe(self, population):
        pass
                    
                
    def quantum_perturbation(self, swarm_id):
        # 量子扰动
        swarm = self.swarms[swarm_id]
        pop_quantum_positions = []
        for _ in range(self.quantum_number):
            quantum_position = swarm['BestPosition'] + (2 * np.random.rand(self.dimension) - 1) * self.quantum_radius
            for k in range(self.dimension):
                if quantum_position[k] > self.max_coordinate:
                    quantum_position[k] = self.max_coordinate
                elif quantum_position[k] < self.min_coordinate:
                    quantum_position[k] = self.min_coordinate
            
            pop_quantum_positions.append(quantum_position)
        return pop_quantum_positions
                
    
    def quantum_update(self, swarm_id, quantum_positions, quantum_fitness):
        swarm = self.swarms[swarm_id]
        for j in range(self.quantum_number):
            if quantum_fitness[j]['f1'] > swarm['BestValue']:
                swarm['BestPosition'] = quantum_positions[j]
                swarm['BestValue'] = quantum_fitness[j]['f1']


    def exclusion_operation(self):
        updated_populations = set()
        for i in range(self.swarm_number - 1):
            for j in range(i + 1, self.swarm_number):
                if i in updated_populations or j in updated_populations:
                    continue
                dist = np.linalg.norm(self.swarms[i]['BestPosition'] - self.swarms[j]['BestPosition'])
                if dist < self.exclusion_limit:
                    if self.swarms[i]['BestValue'] < self.swarms[j]['BestValue']:
                        updated_populations.add(i)
                        self.swarms[i] = self.subpopulation_generator()
                    else:
                        updated_populations.add(j)
                        self.swarms[j] = self.subpopulation_generator()
        updated_populations = list(updated_populations)
        return updated_populations
        
    
    def convergence_check(self):
        # 防聚合操作：检查各子群内部的分散情况
        is_all_converged = 0
        worst_swarm_value = np.inf
        worst_swarm_index = None
        for swarm_id, swarm in enumerate(self.swarms):
            radius = 0
            for j in range(self.pop_size):
                for k in range(self.pop_size):
                    diff = np.abs(swarm['X'][j] - swarm['X'][k])
                    current_distance = np.max(diff)
                    if current_distance > radius:
                        radius = current_distance
            if radius < self.convergence_limit:
                swarm['IsConverged'] = 1
            else:
                swarm['IsConverged'] = 0
            is_all_converged += swarm['IsConverged']
            if swarm['BestValue'] < worst_swarm_value:
                worst_swarm_value = swarm['BestValue']
                worst_swarm_index = swarm_id
        if is_all_converged == self.swarm_number:
            self.swarms[worst_swarm_index] = self.subpopulation_generator()
            
        return worst_swarm_index
    
    def meta_update(self):
        self.change_reaction_mQSO()
    
    def get_past_best_position(self):
        positions = []
        for swarm in self.swarms:
            positions.append(swarm['PbestPosition'])
        return positions
            
    def update_best_value(self, swarm_id, pbest_vals):
        self.swarms[swarm_id]['PbestValue'] = np.array([v['f1'] for v in pbest_vals])  # 更新个体历史最优适应值
        self.swarms[swarm_id]['Gbest_past_environment'] = self.swarms[swarm_id] ['BestPosition']
        best_index = np.argmax(self.swarms[swarm_id]['PbestValue'])
        self.swarms[swarm_id]['BestValue'] = self.swarms[swarm_id]['PbestValue'][best_index]
        self.swarms[swarm_id]['BestPosition'] = self.swarms[swarm_id]['PbestPosition'][best_index].copy()
        self.swarms[swarm_id]['IsConverged'] = 0
        return
    
    
    def iterative_components(self):
        for pop in self.swarms:
            r1 = np.random.rand(self.pop_size, self.dimension)
            r2 = np.random.rand(self.pop_size, self.dimension)
            cognitive_component = self.c1 * r1 * (pop['PbestPosition'] - pop['X'])
            social_component = self.c2 * r2 * (np.tile(pop['BestPosition'], (self.pop_size, 1)) - pop['X'])
            pop['Velocity'] = self.x * (pop['Velocity'] + cognitive_component + social_component)
            
            pop['X'] = pop['X'] + pop['Velocity']
            
            for j in range(self.pop_size):
                for k in range(self.dimension):
                    if pop['X'][j, k] > self.max_coordinate:
                        pop['X'][j, k] = self.max_coordinate
                        pop['Velocity'][j, k] = 0
                    elif pop['X'][j, k] < self.min_coordinate:
                        pop['X'][j, k] = self.min_coordinate
                        pop['Velocity'][j, k] = 0
        
        return
    
        
    def change_reaction(self):
        # 更新 shift severity
        shifts_all = []
        for pop in self.swarms:
            # 检查 Gbest_past_environment 中是否含有 nan
            if not np.isnan(pop['Gbest_past_environment']).any():
                # 计算当前最优位置与上一个环境最优位置间的欧式距离
                shift_val = np.linalg.norm(pop['Gbest_past_environment'] - pop['BestPosition'])
                pop['Shifts'].append(shift_val)
            shifts_all.extend(pop['Shifts'])
        
        if shifts_all:
            self.shift_severity = np.mean(shifts_all)
        self.quantum_radius = self.shift_severity
    
    def check_and_evaluate(self,parameters, budget):
        if len(parameters) < budget:
            return False
        return True
    
    def suggest(self, n_suggestions: None | int = None) -> List[Dict]:
        pass



def plot_statistics(best_error):
    plt.figure(figsize=(12, 8))
    iterations = list(range(len(best_error)))

    plt.plot(iterations, best_error, label='Best Error Before Change')
    plt.xlabel('Iteration')
    plt.ylabel('Best Error distance')
    plt.title('Best Error distance Before Change Over Iterations')
    plt.legend()

    plt.tight_layout()
    plt.savefig('best_error.png')
# ---------------------------
# 示例：如何调用 MQSOAlgorithm
# ---------------------------
if __name__ == '__main__':
    # 创建 MQSOAlgorithm 实例
    n_var = 10  # 1-dimensional

    # Define the required arguments
    task_name = "example_task"
    budget = 10000
    budget_type = "time"
    workloads = list(range(1,100))  # Generate a list from 1 to n
    generator = MovingPeakGenerator(
        task_name=task_name,
        budget=budget,
        budget_type=budget_type,
        workloads=workloads,
        n_var=n_var,
        n_step=12,
        seed=42,
        change_type='oscillatory',
        params={'input_dim': 1}
    )
    problems = generator.generate_benchmarks()
    transfer_problems = TransferProblem(seed=42)
    for problem in problems:
        transfer_problems.add_task(problem)

    mqso = mQSO(pop_size=5, c1=2.05, c2=2.05, seed=0, config={})
    first_flag = True
    break_flag = False
    
    while (transfer_problems.get_unsolved_num()):
        search_space = transfer_problems.get_cur_searchspace()
        mqso.link_task('MPB_{}'.format(transfer_problems.get_cur_task_id()), search_space)
        
        if first_flag:
            mqso.initilization()
            for swarm_id, swarm in enumerate(mqso.swarms):
                samples = swarm['X']
                parameters = [search_space.map_to_design_space(sample) for sample in samples]
                fitness = transfer_problems.f(parameters)
                mqso.update_swarm(swarm_id, swarm, fitness)
            first_flag = False

        while transfer_problems.get_rest_budget():
            start_time = time.time()
            for swarm_id, swarm in enumerate(mqso.swarms):
                pop_quantum_positions = mqso.quantum_perturbation(swarm_id=swarm_id)
                parameters = [search_space.map_to_design_space(sample) for sample in pop_quantum_positions]
                break_flag = mqso.check_and_evaluate(parameters, transfer_problems.get_rest_budget())
                if break_flag:
                    break
                quantum_fitness = transfer_problems.f(parameters)
                mqso.quantum_update(swarm_id=swarm_id, quantum_positions=pop_quantum_positions, quantum_fitness=quantum_fitness)

            if break_flag:
                break
            updated_swarms = mqso.exclusion_operation()
            for swarm_id in updated_swarms:
                parameters = [search_space.map_to_design_space(sample) for sample in mqso.swarms[swarm_id]['X']]
                break_flag = mqso.check_and_evaluate(parameters, transfer_problems.get_rest_budget())
                if break_flag:
                    break
                fitness = transfer_problems.f(parameters)
                mqso.update_swarm(swarm_id=swarm_id, swarm=mqso.swarms[swarm_id], fitness=fitness)
            if break_flag:
                break
            worst_swarm_index = mqso.convergence_check()
            if worst_swarm_index is not None:
                parameters = [search_space.map_to_design_space(sample) for sample in mqso.swarms[worst_swarm_index]['X']]
                break_flag = mqso.check_and_evaluate(parameters, transfer_problems.get_rest_budget())
                if break_flag:
                    break   
                fitness = transfer_problems.f(parameters)
                mqso.update_swarm(swarm_id=worst_swarm_index, swarm=mqso.swarms[worst_swarm_index], fitness=fitness)
            if break_flag:
                break
        
        best_value = max(swarm['BestValue'] for swarm in mqso.swarms)
        samples = np.random.uniform(mqso.min_coordinate, mqso.max_coordinate, size=(10000, mqso.dimension))
        parameters = [search_space.map_to_design_space(sample) for sample in samples]
        transfer_problems.lock()
        fitness = transfer_problems.f(parameters)
        transfer_problems.unlock()
        best_index = np.argmax([fit['f1'] for fit in fitness])
        best_x = parameters[best_index]
        best_y = fitness[best_index]['f1']
        error = best_y - best_value
        mqso.BestErrorBeforeChange.append(error)
        
        print(mqso.BestErrorBeforeChange)
        
        if transfer_problems.get_unsolved_num() == 1:
            break 
        transfer_problems.roll()
        mqso.change_reaction()
        positions = mqso.get_past_best_position()
        for swarm_id, swarm in enumerate(mqso.swarms):
            parameters = [search_space.map_to_design_space(sample) for sample in positions[swarm_id]]
            pbest_vals = transfer_problems.f(parameters)
            mqso.update_best_value(swarm_id=swarm_id, pbest_vals=pbest_vals)
        
    plot_statistics(mqso.BestErrorBeforeChange)





