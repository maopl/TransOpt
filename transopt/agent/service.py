import os
import json
import logging
from pathlib import Path
from typing import Any, Dict

import yaml
from flask import Flask, jsonify, request
from flask_cors import CORS
from log import logger
from openai_connector import Message, OpenAIChat
from transopt.utils.Register import problem_registry

# Assuming OpenAIChat, Message, get_prompt, parse_response are defined correctly

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_prompt(file_name: str) -> str:
    """Reads a prompt from a file."""
    current_dir = Path(__file__).parent
    file_path = current_dir / file_name
    
    with open(file_path, 'r') as file:
        prompt = file.read()
    return prompt


@app.route('/api/generate-yaml', methods=['POST'])
def generate_yaml():
    data = request.json
    user_input = data.get('content', {}).get('text', '')

    # Process the input using the OpenAI API
    system_message = Message(role="system", content=global_prompt)
    user_message = Message(role="user", content=user_input)
    
    if is_first_msg:
        response_content = openai_chat.get_response([system_message, user_message])
    else:
        response_content = openai_chat.get_response([user_message])

    # structured_response = parse_response(response_content)

    return jsonify({"message": response_content}), 200
    # return jsonify(response_content)


@app.route('/api/messages', methods=['POST'])
def handle_message():
    data = request.json
    user_input = data.get('content').get('text')  # 根据前端发送的结构获取用户输入

    response_content = "Processed response here..."  # 替换为实际的处理逻辑

    # 返回处理后的响应给前端
    return jsonify({"message": response_content}), 200


@app.route('/api/report/tasks', methods=['POST'])
def report_send_tasks_information():
    data = request.json
    user_input = data.get('paremeter', '')

    print(user_input)

    # 发送Tasks数据给前端
    current_directory = os.path.dirname(__file__)
    json_file_path = os.path.join(current_directory, 'page_service_data', 'task_information.json')
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return jsonify(data), 200


@app.route('/api/report/charts', methods=['POST'])
def report_update_charts_data():
    data = request.json
    # 从前端得到 taskname
    user_input = data.get('taskname', '')

    # 根据taskname得到对应的json数据
    print(user_input)

    # 发送Tasks数据给前端
    current_directory = os.path.dirname(__file__)
    json_file_path = os.path.join(current_directory, 'page_service_data', 'ReportChartsData.json')
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    # 返回处理后的响应给前端
    return jsonify(data), 200


@app.route('/api/configuration/select_task', methods=['POST'])
def configuration_recieve_tasks():
    data = request.json
    # 从前端得到选择的tasks
    print(data)
    

    # 接收的格式如下
    # [{'name': 'Task1', 'dim': 5, 'obj': 2, 'fidelity': 2}, 
    #  {'name': 'Task2', 'dim': 5, 'obj': 1, 'fidelity': 3}, 
    #  {'name': 'Task3', 'dim': 10, 'obj': 1, 'fidelity': 5}]

    # 返回处理后的响应给前端
    return {"succeed":True}, 200


@app.route('/api/configuration/select_algorithm', methods=['POST'])
def configuration_recieve_algorithm():
    data = request.json
    # 从前端得到选择的算法及其parameters
    print(data)

    # 接收的格式如下
    # {'name': 'Algorithm1', 'parameters': 'Parameter1=1, Parameter2=2'}

    # 返回处理后的响应给前端
    return {"succeed":True}, 200


@app.route('/api/configuration/search_dataset', methods=['POST'])
def configuration_search_dataset():
    data = request.json
    # 从前端得到dataset的搜索条件
    print(data)

    # 接收的格式如下
    # {'name': 0.27, 'dim': 0.62, 'obj': 0.25, 'fidelityName': 0.71, 'fidelity': 0.19}


    data = [
    "dataset1",
    "dataset2",
    "dataset3",
    "dataset4"
    ]

    # 返回处理后的响应给前端
    return jsonify(data), 200


@app.route('/api/configuration/basic_information', methods=['POST'])
def configuration_basic_information():
    data = request.json
    user_input = data.get('paremeter', '')

    print(user_input)
    
    task_names = problem_registry.keys()
    task_data = []
    print(task_names)
    
            
    for name in task_names:
        if  problem_registry[name].get_problem_type() == 'synthetic':
            task_info = {
                "name" : name,
                "anyDim": True,
                "dim" : [],
                'obj':[1],
                'fidelity':[]
            }
        else:
            obj_num =  problem_registry[name].get_objectives()
            dim = len(problem_registry[name].get_configuration_space().keys())
            task_info = {
                "name" : name,
                "anyDim": False,
                "dim" : [dim],
                'obj':[obj_num],
                'fidelity':[]
            }
        task_data.append(task_info)

    # 发送Tasks数据给前端
    current_directory = os.path.dirname(__file__)
    json_file_path = os.path.join(current_directory, 'page_service_data', 'configuration_basic.json')
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return jsonify(data), 200


@app.route('/api/configuration/begin', methods=['POST'])
def configuration_begin():  
    data = request.json
    # 从前端得到选择的dataset，并开始实验
    print(data)

    # 接收的格式如下
    # ['dataset1', 'dataset2', 'dataset3', 'dataset4']

    # 返回处理后的响应给前端
    return {"succeed":True}, 200


@app.route('/api/comparison/tasks', methods=['POST'])
def comparisno_tasks():
    data = request.json
    user_input = data.get('paremeter', '')

    print(user_input)
    # comparison 页面可供选择的 Tasks
    data = ["Task1", "Task2", "Task3"]

    # 返回处理后的响应给前端
    return jsonify(data), 200


@app.route('/api/comparison/choose_task', methods=['POST'])
def comparisno_choose_tasks():
    data = request.json
    user_input = data.get('task', '')

    # 选择的Task，返回对应可选的Algorithm
    print(user_input)

    if user_input == "Task1":
        data = ["Algorithm1", "Algorithm2", "Algorithm3"]
    elif user_input == "Task2":
        data = ["BO", "MTBO"]
    else:
        data = ["Algorithm4"]

    # 返回处理后的响应给前端
    return jsonify(data), 200


@app.route('/api/comparison/charts', methods=['POST'])
def comparison_update_charts_data():
    data = request.json
    # 从前端得到 taskname 和 Algorithms
    selected_task = data.get('taskname', '')
    print(selected_task)
    selected_algorithm = data.get('selectedAlgorithms', '')
    print(selected_algorithm)


    # 发送charts的数据
    current_directory = os.path.dirname(__file__)
    json_file_path = os.path.join(current_directory, 'page_service_data', 'ComparisonChartsData.json')
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    # 返回处理后的响应给前端
    return jsonify(data), 200


if __name__ == '__main__':
    global_prompt = get_prompt("prompt")
    openai_chat = OpenAIChat()
    is_first_msg = True

    app.run(debug=True)

    