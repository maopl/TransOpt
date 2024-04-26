
class Config:
    DEBUG = True
    OPENAI_API_KEY = "sk-eGYDsI7kGLAVM9bs2585D337E51b48FbA30d88B0Fa8a1571"
    OPENAI_URL = "https://aihubmix.com/v1"




class RunningConfig:
    def __init__(self):
        self.tasks = None
        self.optimizer = None
        self.metadata = {'SpaceRefiner':[], 'Sampler':[], 'ACF':[], 'Pretrain':[], 'Model':[], 'Normalizer':[]}
        
        
    def set_tasks(self, tasks):
        self.tasks = tasks
        
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_metadata(self, metadata):
        self.metadata[metadata['object']] = metadata['datasets']

