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
    
    all_tasks = [task_info['additional_config'] for task_info in services.get_all_tasks()]
    
    # with open("./page_service_data/task_information.json", "w") as file:
        # json.dump(all_tasks, file)
    return jsonify(all_tasks), 200


@app.route("/api/report/charts", methods=["POST"])
def report_update_charts_data():
    data = request.json
    user_input = data.get("taskname", "")

    charts = services.get_report_charts(user_input)
    return jsonify(charts), 200


@app.route("/api/configuration/select_task", methods=["POST"])
def configuration_recieve_tasks():
    tasks_info = request.json
    
    services.receive_tasks(tasks_info)
    return {"succeed": True}, 200


@app.route("/api/configuration/select_algorithm", methods=["POST"])
def configuration_recieve_algorithm():
    optimizer_info = request.json

    services.receive_optimizer(optimizer_info)
    return {"succeed": True}, 200


@app.route("/api/configuration/basic_information", methods=["POST"])
def configuration_basic_information():
    data = request.json
    user_input = data.get("paremeter", "")

    task_data = services.get_modules()
    # with open('transopt/agent/page_service_data/configuration_basic.json', 'r') as file:
    #     data = json.load(file)
    return jsonify(task_data), 200


@app.route("/api/configuration/dataset", methods=["POST"])
def configuration_dataset():
    metadata_info = request.json
    
    services.select_dataset(metadata_info)
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
        data = ["1","2","3"]
        return jsonify(datasets), 200
    except Exception as e:
        logger.error(f"Error in searching dataset: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/configuration/run", methods=["POST"])
def configuration_run():
    run_info = request.json
    print(run_info)

    try:
        services.run_optimize(seeds_info = run_info['Seeds'])
    except:
        # raise Exception("Error in running the optimization")
        return {"isSucceed": False}, 200

    # 返回处理后的响应给前端
    return {"isSucceed": True}, 200


@app.route("/api/comparison/tasks", methods=["POST"])
def comparisno_tasks():
    data = request.json
    user_input = data.get("paremeter", "")

    print(user_input)
    data = ["Task1", "Task2", "Task3"]
    return jsonify(data), 200


@app.route("/api/comparison/choose_task", methods=["POST"])
def comparisno_choose_tasks():
    data = request.json
    user_input = data.get("task", "")

    print(user_input)

    if user_input == "Task1":
        data = ["Algorithm1", "Algorithm2", "Algorithm3"]
    elif user_input == "Task2":
        data = ["BO", "MTBO"]
    else:
        data = ["Algorithm4"]
    return jsonify(data), 200


@app.route("/api/comparison/charts", methods=["POST"])
def comparison_update_charts_data():
    data = request.json
    selected_task = data.get("taskname", "")
    print(selected_task)
    selected_algorithm = data.get("selectedAlgorithms", "")
    print(selected_algorithm)

    current_directory = os.path.dirname(__file__)
    json_file_path = os.path.join(
        current_directory, "page_service_data", "ComparisonChartsData.json"
    )
    with open(json_file_path, "r") as file:
        data = json.load(file)
    return jsonify(data), 200


if __name__ == "__main__":
    app.run(debug=services.config.DEBUG)
