"""hyperparameters.py: Helper class to process images
"""
__author__ = "Mihai Matei"
__license__ = "BSD"
__email__ = "mihai.matei@my.fmi.unibuc.ro"

import itertools
from shutil import copyfile
import uuid

from .model import Model
from .features import ModelDataSet


class HyperParamsLookup:
    def __init__(self, model: Model, performance_callback):
        self._model = model
        self._best_params = None
        self._best_history = None
        self._best_checkpoint = './best_model_' + str(uuid.uuid4()) + '.save'
        self._history = []
        self._best_performance = 0
        self._performance_callback = performance_callback

    def grid_search(self, data: ModelDataSet, log=False, **hyper_space):
        hyper_keys = hyper_space.keys()
        hyper_values = hyper_space.values()

        for hyper_params in itertools.product(*hyper_values):
            model_init = {}
            for hyper_key, hyper_val in zip(hyper_keys, hyper_params):
                model_init[hyper_key] = hyper_val

            self._model.__init__(**model_init)
            history = self._model.train(data)

            self._history.append((model_init.copy(), history))
            perf = self._performance_callback(history)
            if log:
                print("Hyperparameters: {0}\nResults: {1}".format(model_init, perf))
            if perf > self._best_performance:
                self._best_performance = perf
                self._best_params = model_init.copy()
                self._best_history = history
                if self._model.checkpoint() is not None:
                    copyfile(self._model.checkpoint(), self.best_checkpoint)

            self._model.destroy()

    @property
    def best_params(self):
        return self._best_params

    @property
    def best_history(self):
        return self._best_history

    @property
    def history(self):
        return self._history

    @property
    def best_checkpoint(self):
        return self._best_checkpoint
