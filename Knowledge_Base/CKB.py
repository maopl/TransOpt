import numpy as np


class KnowledgeBase():
    def __init__(self):
        self.name = []
        self.type = []
        self.model = []
        self.x = []
        self.y = []
        self.len = 0
        self.current_task_x = []
        self.metrics = []
        self.prior = []
        self.match_id = []

    def add(self, fun_name, type, x, y, model=None, prior=None, metric=None, match_id = None):
        if fun_name not in self.name:
            self.name.append(fun_name)
            self.type.append(type)
            self.model.append(model)
            self.prior.append(prior)
            self.metrics.append(metric)
            self.x.append([x])
            self.y.append([y])
            self.match_id.append(match_id)
            self.len += 1
        else:
            ind = self.name.index(fun_name)
            self.x[ind] = [x]
            self.y[ind] = [y]
            self.type[ind] = type
            self.model[ind] = model
            self.prior[ind] = prior
            self.metrics[ind] = metric
            self.match_id[ind] = match_id

    def update(self, KB_id, x, y, model=None, prior=None, metric=None, match_id = None):
        self.x[KB_id].append(x)
        self.y[KB_id].append(y)
        self.type[KB_id] = type
        self.model[KB_id] = model
        self.prior[KB_id] = prior
        self.metrics[KB_id] = metric
        self.match_id[KB_id] = match_id



    def delete(self, KB_id):
        self.name.pop(KB_id)
        self.type.pop(KB_id)
        self.x.pop(KB_id)
        self.y.pop(KB_id)
        self.model.pop(KB_id)
        self.prior.pop(KB_id)
        self.metrics.pop(KB_id)
        self.match_id.pop(KB_id)
        self.len -= 1

    def update_prior(self, KB_id, prior):
        self.prior[KB_id] = prior

    def search_task(self, fun_name):
        if fun_name not in self.name:
            return None
        else:
            return self.name.index(fun_name)
