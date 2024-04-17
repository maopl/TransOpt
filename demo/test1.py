import logging
from pathlib import Path
from typing import Any, Dict

import yaml
from flask import Flask, jsonify, request
from flask_cors import CORS
from .log import logger
from .openai_connector import Message, OpenAIChat

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


@app.route('/api/task_button', methods=['POST'])
def generate_yaml():
    data = request.json
    user_input = data.get('paremeter', '')

    print(user_input)

    return jsonify({"message": 'Done'}), 200
    # return jsonify(response_content)


@app.route('/api/messages', methods=['POST'])
def handle_message():
    data = request.json
    user_input = data.get('content').get('text')  # 根据前端发送的结构获取用户输入

    response_content = "Processed response here..."  # 替换为实际的处理逻辑

    # 返回处理后的响应给前端
    return jsonify({"message": response_content}), 200

if __name__ == '__main__':
    app.run(debug=True)

    