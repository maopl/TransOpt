import json
import os

from flask import Flask, jsonify, request
from flask_cors import CORS
from services import Services

from transopt.agent.registry import *
from transopt.utils.log import logger

app = Flask(__name__)
CORS(app)
services = Services()
print("services")

@app.route("/api/generate-yaml", methods=["POST"])
def generate_yaml():
    try:
        data = request.json
        user_input = data.get("content", {}).get("text", "")
        response_content = services.chat(user_input)
        return jsonify({"message": response_content}), 200
    except Exception as e:
        logger.error(f"Error in generating YAML: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/report/tasks", methods=["POST"])
def report_send_tasks_information():
    data = request.json
    user_input = data.get("paremeter", "")

    print(user_input)

    # 发送Tasks数据给前端
    current_directory = os.path.dirname(__file__)
    json_file_path = os.path.join(
        current_directory, "page_service_data", "task_information.json"
    )
    with open(json_file_path, "r") as file:
        data = json.load(file)
    return jsonify(data), 200


@app.route("/api/report/charts", methods=["POST"])
def report_update_charts_data():
    data = request.json
    # 从前端得到 taskname
    user_input = data.get("taskname", "")

    # 根据taskname得到对应的json数据
    print(user_input)

    # 发送Tasks数据给前端
    current_directory = os.path.dirname(__file__)
    json_file_path = os.path.join(
        current_directory, "page_service_data", "ReportChartsData.json"
    )
    with open(json_file_path, "r") as file:
        data = json.load(file)
    # 返回处理后的响应给前端
    return jsonify(data), 200


@app.route("/api/configuration/select_task", methods=["POST"])
def configuration_recieve_tasks():
    data = request.json
    # 从前端得到选择的tasks
    print(data)

    # 接收的格式如下
    # [{'name': 'Task1', 'dim': 5, 'obj': 2, 'fidelity': 2},
    #  {'name': 'Task2', 'dim': 5, 'obj': 1, 'fidelity': 3},
    #  {'name': 'Task3', 'dim': 10, 'obj': 1, 'fidelity': 5}]

    # 返回处理后的响应给前端
    return {"succeed": True}, 200


@app.route("/api/configuration/select_algorithm", methods=["POST"])
def configuration_recieve_algorithm():
    data = request.json
    # 从前端得到选择的算法及其parameters
    print(data)

    # 接收的格式如下
    # {'name': 'Algorithm1', 'parameters': 'Parameter1=1, Parameter2=2'}

    # 返回处理后的响应给前端
    return {"succeed": True}, 200


@app.route("/api/configuration/basic_information", methods=["POST"])
def configuration_basic_information():
    data = request.json
    user_input = data.get("paremeter", "")

    print(user_input)

    task_names = list(g_problem_registry.keys())
    task_data = []
    algoprithm_data = []

    for name in task_names:
        if g_problem_registry[name].get_problem_type() == "synthetic":
            task_info = {
                "name": name,
                "anyDim": True,
                "dim": [],
                "obj": [1],
                "fidelity": [],
            }
        else:
            obj_num = g_problem_registry[name].get_objectives()
            dim = len(g_problem_registry[name].get_configuration_space().keys())
            task_info = {
                "name": name,
                "anyDim": False,
                "dim": [dim],
                "obj": [obj_num],
                "fidelity": [],
            }
        task_data.append(task_info)

    # 发送Tasks数据给前端
    current_directory = os.path.dirname(__file__)
    json_file_path = os.path.join(
        current_directory, "page_service_data", "configuration_basic.json"
    )
    with open(json_file_path, "r") as file:
        data = json.load(file)
    return jsonify(data), 200


@app.route("/api/configuration/dataset", methods=["POST"])
def configuration_dataset():
    data = request.json
    # 从前端得到选择的dataset，并开始实验
    print(data)

    # 接收的格式如下
    # ['dataset1', 'dataset2', 'dataset3', 'dataset4']

    # 返回处理后的响应给前端
    return {"succeed": True}, 200


@app.route("/api/configuration/search_dataset", methods=["POST"])
def configuration_search_dataset():
    try:
        data = request.json

        dataset_name = data["task_name"]
        dataset_info = {
            "num_variables": data["num_variables"],
            "num_objectives": data["num_objectives"],
            "variables": [
                {"name": var_name} for var_name in data["variables_name"].split(",")
            ],
        }
        datasets = services.search_dataset(dataset_name, dataset_info)

        return jsonify(datasets), 200
    except Exception as e:
        logger.error(f"Error in searching dataset: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/configuration/run", methods=["POST"])
def configuration_run():
    data = request.json
    # 从前端得到开始实验的信息
    print(data.message)

    # 返回处理后的响应给前端
    return {"succeed": True}, 200


@app.route("/api/comparison/tasks", methods=["POST"])
def comparisno_tasks():
    data = request.json
    user_input = data.get("paremeter", "")

    print(user_input)
    # comparison 页面可供选择的 Tasks
    data = ["Task1", "Task2", "Task3"]

    # 返回处理后的响应给前端
    return jsonify(data), 200


@app.route("/api/comparison/choose_task", methods=["POST"])
def comparisno_choose_tasks():
    data = request.json
    user_input = data.get("task", "")

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


@app.route("/api/comparison/charts", methods=["POST"])
def comparison_update_charts_data():
    data = request.json
    # 从前端得到 taskname 和 Algorithms
    selected_task = data.get("taskname", "")
    print(selected_task)
    selected_algorithm = data.get("selectedAlgorithms", "")
    print(selected_algorithm)

    # 发送charts的数据
    current_directory = os.path.dirname(__file__)
    json_file_path = os.path.join(
        current_directory, "page_service_data", "ComparisonChartsData.json"
    )
    with open(json_file_path, "r") as file:
        data = json.load(file)
    # 返回处理后的响应给前端
    return jsonify(data), 200


if __name__ == "__main__":
    app.run(debug=services.config.DEBUG)
