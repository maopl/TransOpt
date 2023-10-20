import warnings

from External.tpe.optimizer.random_search import RandomSearch
from External.tpe.optimizer.tpe_optimizer import TPEOptimizer


warnings.filterwarnings("ignore")

opts = {
    "tpe": TPEOptimizer,
    "random_search": RandomSearch,
}
