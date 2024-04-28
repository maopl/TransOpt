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
    # try:
    data = request.json
    user_input = data.get("content", {}).get("text", "")
    response_content = services.chat(user_input)
    return jsonify({"message": response_content}), 200
    # except Exception as e:
    #     logger.error(f"Error in generating YAML: {e}")
    #     return jsonify({"error": str(e)}), 500


@app.route("/api/report/tasks", methods=["POST"])
def report_send_tasks_information():
    data = request.json
    user_input = data.get("paremeter", "")
    
    all_tasks = [task_info['additional_config'] for task_info in services.get_experiment_datasets()]
    
    # with open("./page_service_data/task_information.json", "w") as file:
        # json.dump(all_tasks, file)
    return jsonify(all_tasks), 200


@app.route("/api/report/charts", methods=["POST"])
def report_update_charts_data():
    data = request.json
    user_input = data.get("taskname", "")
    # 其他的图, 数据格式和以前一样 {"RadarData":..., "BarData":..., "ScatterData":...}
    charts = services.get_report_charts(user_input)
    return jsonify(charts), 200


@app.route("/api/report/trajectory", methods=["POST"])
def report_update_trajectory_data():
    data = request.json
    user_input = data.get("taskname", "")
    # trajectory, 数据格式和以前一样 {"TrajectoryData":...}
    charts = services.get_report_charts(user_input)
    return jsonify(charts), 200


@app.route("/api/configuration/select_task", methods=["POST"])
def configuration_recieve_tasks():
    tasks_info = request.json
    # try:
    services.receive_tasks(tasks_info) 
    # except Exception as e:
    #     logger.error(f"Error in searching dataset: {e}")
    #     return jsonify({"error": str(e)}), 500
    
    return {"succeed": True}, 200


@app.route("/api/configuration/select_algorithm", methods=["POST"])
def configuration_recieve_algorithm():
    optimizer_info = request.json
    print(optimizer_info)
    # optimizer_info = {'SpaceRefiner': 'default', 
    #                   'SpaceRefinerParameters': '', 
    #                   'SpaceRefinerDataSelector': 'default', 
    #                   'SpaceRefinerDataSelectorParameters': '', 
    #                   'Sampler': 'default', 
    #                   'SamplerParameters': '', 
    #                   'SamplerInitNum': '11',
    #                   'SamplerDataSelector': 'default', 
    #                   'SamplerDataSelectorParameters': '', 
    #                   'Pretrain': 'default', 
    #                   'PretrainParameters': '', 
    #                   'PretrainDataSelector': 'default', 
    #                   'PretrainDataSelectorParameters': '', 
    #                   'Model': 'default', 
    #                   'ModelParameters': '', 
    #                   'ModelDataSelector': 'default', 
    #                   'ModelDataSelectorParameters': '', 
    #                   'ACF': 'default', 
    #                   'ACFParameters': '', 
    #                   'ACFDataSelector': 'default', 
    #                   'ACFDataSelectorParameters': '', 
    #                   'Normalizer': 'default', 
    #                   'NormalizerParameters': '', 
    #                   'NormalizerDataSelector': 'default', 
    #                   'NormalizerDataSelectorParameters': ''}
    try:
        services.receive_optimizer(optimizer_info)
    except Exception as e:
        logger.error(f"Error in searching dataset: {e}")
        return jsonify({"error": str(e)}), 500
    
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
    # print(metadata_info)
    # metadate_info = {
    #     "object": "Space refiner",
    #     "datasets": ["dataset1", "dataset2]
    # }
    try:
        services.select_dataset(metadata_info)
    except Exception as e:
        logger.error(f"Error in searching dataset: {e}")
        return jsonify({"error": str(e)}), 500
    
    return {"succeed": True}, 200


@app.route("/api/configuration/search_dataset", methods=["POST"])
def configuration_search_dataset():
    try:
        data = request.json

        dataset_name = data["task_name"]
        if data['search_method'] == 'Fuzzy' or 'Hash':
            dataset_info = {}
        elif data['search_method'] == 'LSH':
            dataset_info = {
                "num_variables": data["num_variables"],
                "num_objectives": data["num_objectives"],
                "variables": [
                    {"name": var_name} for var_name in data["variables_name"].split(",")
                ],
            }
        else:
            pass
        datasets = services.search_dataset(data['search_method'], dataset_name, dataset_info)

        return jsonify(datasets), 200
    except Exception as e:
        logger.error(f"Error in searching dataset: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/configuration/delete_dataset", methods=["POST"])
def configuration_delete_dataset():
    metadata_info = request.json
    datasets = metadata_info["datasets"]
    print(datasets)
    # 删除选中的数据集
    return {"succeed": True}, 200


@app.route("/api/configuration/run", methods=["POST"])
def configuration_run():
    run_info = request.json
    print(run_info)

    # try:
    services.run_optimize(seeds_info = run_info['Seeds'])
    # except Exception as e:
    #     logger.error(f"Error in optimization: {e}")
    #     return jsonify({"error": str(e)}), 500

    # 返回处理后的响应给前端
    return {"isSucceed": True}, 200


@app.route("/api/comparison/selections", methods=["POST"])
def comparison_send_selections():
    info = request.json
    # Comparison初始化时，请求可选择的搜索选项
    current_directory = os.path.dirname(__file__)
    json_file_path = os.path.join(
        current_directory, "page_service_data", "ComparisonSelection.json"
    )
    with open(json_file_path, "r") as file:
        data = json.load(file)
        
    
    return jsonify(data), 200


@app.route("/api/comparison/choose_task", methods=["POST"])
def comparison_choose_tasks():
    conditions = request.json
    print(conditions)
    # 根据选择的搜索条件，筛选出对应的任务进行比较，返回比较的图

    datasets = services.comparision_search(conditions)
    # current_directory = os.path.dirname(__file__)
    # json_file_path = os.path.join(
    #     current_directory, "page_service_data", "ComparisonChartsData.json"
    # )
    # with open(json_file_path, "r") as file:
    #     data = json.load(file)
    return jsonify(data), 200


if __name__ == "__main__":
    app.run(debug=services.config.DEBUG)
