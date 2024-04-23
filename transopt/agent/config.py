
class Config:
    DEBUG = True
    OPENAI_API_KEY = "sk-1234567890abcdef1234567890abcdef"
    OPENAI_URL = "https://api.openai.com/v1/engines/davinci-codex/completions"




class RunningConfig:
    def __init__(self):
        self.tasks = None
        self.optimizer = None
        self.metadata = None
        
        
    def set_tasks(self, tasks):
        self.tasks = tasks
        
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_metadata(self, metadata):
        self.optimizer = metadata

