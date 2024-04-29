import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import yaml
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel

from transopt.datamanager.manager import DataManager
from transopt.utils.log import logger
from transopt.agent.registry import *

class Message(BaseModel):
    """Model for LLM messages"""

    role: str  # The role of the message author (system, user, assistant, or function).
    content: Optional[Union[str, List[Dict]]] = None  # The message content.
    tool_call_id: Optional[str] = None  # ID for the tool call response
    name: Optional[str] = None  # Name of the tool or function, if applicable
    metrics: Dict[str, Any] = {}  # Metrics for the message.
    

    def get_content_string(self) -> str:
        """Returns the content as a string."""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            return json.dumps(self.content)
        return ""

    def to_dict(self) -> Dict[str, Any]:
        _dict = self.model_dump(exclude_none=True, exclude={"metrics"})
        # Manually add the content field if it is None
        if self.content is None:
            _dict["content"] = None
        return _dict

    def log(self, level: Optional[str] = None):
        """Log the message to the console."""
        _logger = getattr(logger, level or "debug")
        
        _logger(f"============== {self.role} ==============")
        message_detail = f"Content: {self.get_content_string()}"
        if self.tool_call_id:
            message_detail += f", Tool Call ID: {self.tool_call_id}"
        if self.name:
            message_detail += f", Name: {self.name}"
        _logger(message_detail)


class OpenAIChat:
    history: List[Message]

    def __init__(
        self,
        api_key,
        model="gpt-3.5-turbo",
        base_url="https://api.openai.com/v1",
        client_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key 
        self.client_kwargs = client_kwargs or {}

        self.prompt = self._get_prompt()
        self.is_first_msg = True
        
        self.history = []

        self.data_manager = DataManager()

    def _get_prompt(self):
        """Reads a prompt from a file."""
        current_dir = Path(__file__).parent
        file_path = current_dir / "prompt"
        with open(file_path, "r") as file:
            return file.read()
        
    @property
    def client(self):
        """Lazy initialization of the OpenAI client."""
        from openai import OpenAI
        return OpenAI(
            api_key=self.api_key, base_url=self.base_url, **self.client_kwargs
        )

    def invoke_model(self, messages: List[Dict]) -> ChatCompletion:
        self.history.extend(messages)
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_all_datasets",
                    "description": "Get a list of all available datasets",
                    "parameters": {},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_dataset_info",
                    "description": "Get detailed information about a dataset",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "dataset_name": {
                                "type": "string",
                                "description": "The name of the dataset",
                            },
                        },
                        "required": ["dataset_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_all_problems",
                    "description": "Get all optimization problems that our system supoorts",
                    "parameters": {},
                },
            },
            
            {
                "type": "function",
                "function": {
                    "name": "get_optimizer_components",
                    "description": "Get all components of optimizer that our system supoorts, and show the techniques that can be used in each component",
                    "parameters": {},
                },
            },
                        
            {
                "type": "function",
                "function": {
                    "name": "set_optimization_problem",
                    "description": "This function is called when a user wants to perform an optimization or find an optimal solution for a given task.\
                        It facilitates the optimization process by requiring the user to specify certain parameters that\
                        define the 'workload' and 'budget' available for the task.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "problem_name": {
                                "type": "string",
                                "description": "The name of the optimization problem",
                            },
                            "workload": {
                                "type": "integer",
                                "description": "The number of workload",
                            },
                            "budget": {
                                "type": "integer",
                                "description": "The number of budget to do function evaluations",
                            },
                        },
                        "required": ["problem_name", "workload", "budget"],
                    },
                },
            },
        ]
                
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        # Process tool calls if there are any
        if tool_calls:
            self.history.append(response_message)
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                function_response = self.call_manager_function(function_name, **function_args)
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": function_response
                }
                self.history.append(tool_message)
                
            # Refresh the model with the function response and get a new response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
            )
        
        self.history.append(response.choices[0].message) 
        logger.debug(f"Response: {response.choices[0].message.content}")
        return response

    def get_response(self, user_input) -> str:
        logger.debug("---------- OpenAI Response Start ----------")
        user_message = {"role": "user", "content": user_input}
        logger.debug(f"User: {user_input}")
        messages = [user_message]

        if self.is_first_msg:
            system_message = {"role": "system", "content": self.prompt}
            messages.insert(0, system_message)
            self.is_first_msg = False
        else:
            system_message = {"role": "system", "content": "Don't tell me which function to use, just call it. Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous"}
            messages.insert(0, system_message)
            

        response = self.invoke_model(messages)
        logger.debug(f"Assistant: {response.choices[0].message.content}")
        logger.debug("---------- OpenAI Response End ----------")
        return response.choices[0].message.content 
    
    def call_manager_function(self, function_name, **kwargs):
        available_functions = {
            "get_all_datasets": self.data_manager.get_all_datasets,
            "get_all_problems": self.get_all_problems,
            "get_optimizer_components": self.get_optimizer_components,
            "get_dataset_info": lambda: self.data_manager.get_dataset_info(kwargs['dataset_name']),
            "set_optimization_problem": lambda: self.set_optimization_problem(kwargs['problem_name', 'workload', 'budget']),
            
        }
        function_to_call = available_functions[function_name]
        return json.dumps({"result": function_to_call()})
    
    def _initialize_modules(self):
        import transopt.benchmark.synthetic
        # import transopt.benchmark.CPD
        import transopt.optimizer.acquisition_function
        import transopt.optimizer.model
        import transopt.optimizer.pretrain
        import transopt.optimizer.refiner
        import transopt.optimizer.sampler

    def get_all_problems(self):
        tasks_info = []

        # tasks information
        task_names = problem_registry.list_names()
        for name in task_names:
            if problem_registry[name].problem_type == "synthetic":
                num_obj = problem_registry[name].num_objectives
                num_var = problem_registry[name].num_variables
                task_info = {
                    "name": name,
                    "problem_type": "synthetic",
                    "anyDim": "True",
                    'num_vars': [],
                    "num_objs": [1],
                    "workloads": [],
                    "fidelity": [],
                }
            else:
                num_obj = problem_registry[name].num_objectives
                num_var = problem_registry[name].num_variables
                fidelity = problem_registry[name].fidelity
                workloads = problem_registry[name].workloads
                task_info = {
                    "name": name,
                    "problem_type": "synthetic",
                    "anyDim": False,
                    "num_vars": [num_var],
                    "num_objs": [num_obj],
                    "workloads": [workloads],
                    "fidelity": [fidelity],
                }
            tasks_info.append(task_info)
        return tasks_info
    
    def get_optimizer_components(self):
        basic_info = {}

        selector_info = []
        model_info = []
        sampler_info = []
        acf_info = []
        pretrain_info = []
        refiner_info = []
        normalizer_info = []
        
        # tasks information
        sampler_names = sampler_registry.list_names()
        for name in sampler_names:
            sampler_info.append(name)
        basic_info["Sampler"] = ','.join(sampler_info)

        refiner_names = space_refiner_registry.list_names()
        for name in refiner_names:
            refiner_info.append(name)
        basic_info["SpaceRefiner"] = ','.join(refiner_info)

        pretrain_names = pretrain_registry.list_names()
        for name in pretrain_names:
            pretrain_info.append(name)
        basic_info["Pretrain"] = ','.join(pretrain_info)

        model_names = model_registry.list_names()
        for name in model_names:
            model_info.append(name)
        basic_info["Model"] = ','.join(model_info)

        acf_names = acf_registry.list_names()
        for name in acf_names:
            acf_info.append(name)
        basic_info["ACF"] = ','.join(acf_info)

        selector_names = selector_registry.list_names()
        for name in selector_names:
            selector_info.append(name)
        basic_info["DataSelector"] = ','.join(selector_info)
        
        normalizer_names = selector_registry.list_names()
        for name in normalizer_names:
            normalizer_info.append(name)
        basic_info["Normalizer"] = ','.join(normalizer_info)
        
        
        return basic_info
    
    def set_optimization_problem(self, problem_name, workload, budget):
        problem = problem_registry[problem_name]
        problem.set_workload(workload)
        problem.set_budget(budget)
        
        problem_info = {}
        w = [int(item) for item in workload.split(",")]
        if problem_name in problem_registry:
            problem_info[problem_name] = {
                'budget': budget,
                'workload': w,
                'budget_type': 'Num_FEs',
            }

        self.running_config.set_tasks(problem_info)
        return "Succeed"
