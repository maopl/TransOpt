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




global_prompt = get_prompt("prompt")
openai_chat = OpenAIChat()
is_first_msg = True

app.run(debug=True)