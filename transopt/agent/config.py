
class ChatbotConfig:
    DEBUG = True
    OPENAI_API_KEY = "sb-d19dea274ed2ed0b2539aa75920829d2d419a03504802d7c"
    OPENAI_URL = "https://api.openai-sb.com/v1/"



import json
import os

class Configer:
    _instance = None
    _init = False  # 用于保证初始化代码只运行一次
    CONFIG_FILE = os.path.join("config", "running_config.json")

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Configer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        # 确保config目录存在
        os.makedirs("config", exist_ok=True)
        self._load_config()
        
    def _load_config(self):
        """Load config from JSON file or initialize defaults"""
        if os.path.exists(self.CONFIG_FILE):
            with open(self.CONFIG_FILE, 'r') as f:
                config = json.load(f)
                self.tasks = config.get('tasks', [])
                self.optimizer = config.get('optimizer', {
                    'SearchSpace': {'type': None, 'auxiliaryData': [], 'autoSelect': False},
                    'Initialization': {'type': 'random', 'InitNum': 11, 'auxiliaryData': [], 'autoSelect': False},
                    'AcquisitionFunction': {'type': 'EI', 'auxiliaryData': [], 'autoSelect': False},
                    'Pretrain': {'type': None, 'auxiliaryData': [], 'autoSelect': False},
                    'Model': {'type': 'GP', 'auxiliaryData': [], 'autoSelect': False},
                    'Normalizer': {'type': 'Standard', 'auxiliaryData': [], 'autoSelect': False}
                })
                self.seeds = config.get('seeds', '42')
                self.remote = config.get('remote', False)
                self.server_url = config.get('server_url', '')
                self.exp_name = config.get('experimentName', '')
                self.exp_description = config.get('experimentDescription', '')
        else:
            self.tasks = []
            self.optimizer = {
                'SearchSpace': {'type': None, 'auxiliaryData': [], 'autoSelect': False},
                'Initialization': {'type': 'random', 'InitNum': 11, 'auxiliaryData': [], 'autoSelect': False},
                'AcquisitionFunction': {'type': 'EI', 'auxiliaryData': [], 'autoSelect': False},
                'Pretrain': {'type': None, 'auxiliaryData': [], 'autoSelect': False},
                'Model': {'type': 'GP', 'auxiliaryData': [], 'autoSelect': False},
                'Normalizer': {'type': 'Standard', 'auxiliaryData': [], 'autoSelect': False}}
            self.seeds = '42'
            self.remote = False
            self.server_url = ''
            self.exp_name = ''
            self.exp_description = ''
            self._save_config()

    def _save_config(self):
        """Save current config to JSON file"""
        config = {
            'tasks': self.tasks,
            'optimizer': self.optimizer,
            # 'metadata': self.metadata,
            'seeds': self.seeds,
            'remote': self.remote,
            'server_url': self.server_url,
            'experimentName': self.exp_name,
            'experimentDescription': self.exp_description
        }
        with open(self.CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
            
            
    def set_configuration(self, config_info):
        """Set configuration from config_info dict"""
        # Set tasks if present        
        self.tasks = config_info['tasks']
        
        for i in config_info['optimizer']:
            self.optimizer[i['name']] = {k:v for k,v in i.items() if k != 'name'}
        # self.metadata = config_info['datasets']
        self.seeds = config_info['Seeds']
        self.remote = config_info['Remote']
        self.server_url = config_info['server_url']
        self.exp_name = config_info['experimentName']
        self.exp_description = config_info['experimentDescription']
        
        self._save_config()
        

    def get_configuration(self):
        """Get current configuration"""
        with open(self.CONFIG_FILE, 'r') as f:
            config = json.load(f)
            return {
                'tasks': config.get('tasks'),
                'optimizer': config.get('optimizer'),
                'seeds': config.get('seeds', '42'),
                'remote': config.get('remote', False),
                'server_url': config.get('server_url', ''),
                'experimentName': config.get('experimentName', ''),
                'experimentDescription': config.get('experimentDescription', '')
            }
