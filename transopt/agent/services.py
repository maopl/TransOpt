from transopt.agent.registry import *


from transopt.agent.chat.openai_connector import (
    Message,
    OpenAIChat,
    get_prompt
)
from transopt.agent.config import Config
from transopt.datamanager.manager import DataManager


class Services:
    def __init__(self):
        self.config = Config()
        self.data_manager = DataManager()
        
        self.openai_chat = OpenAIChat(self.config)
        self.prompt = get_prompt()
        self.is_first_msg = True
        self.initialize_modules()

    def chat(self, user_input):
        system_message = Message(role="system", content=self.prompt)
        user_message = Message(role="user", content=user_input)
        
        if self.is_first_msg:
            response_content = self.openai_chat.get_response([system_message, user_message])
        else:
            response_content = self.openai_chat.get_response([user_message])
        
        return response_content
        
    def search_dataset(self, dataset_name, dataset_info):
        return list(self.data_manager.get_similar_datasets(dataset_name, dataset_info))
    
    
    
    def run_optimize(self):
        pass
    
    def initialize_modules(self):
        import transopt.benchmark.synthetic
        import transopt.optimizer.acquisition_function
        import transopt.optimizer.model
        import transopt.optimizer.pretrain
        import transopt.optimizer.sampler
        import transopt.optimizer.refiner
        
