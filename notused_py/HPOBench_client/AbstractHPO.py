import numpy as np


class HPO():
    def __init__(self,
        input_dim = 1,
        bounds = None,
        RX = None,
        Seed = None,
    ):

        self.xdim = input_dim
        self.RX = np.array(RX)

        self.query_num = 0

        np.random.seed(Seed)
        if bounds is None:
            self.bounds = np.array([[-1.0] * self.xdim, [1.0] * self.xdim])
        else:
            self.bounds = bounds


    def launch(self):

        raise NotImplemented
