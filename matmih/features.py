"""features.py: Helper classes to hold dataset information
Process a data set to be used with a Model class
The features and targets are computed as numpy darrays
"""
__author__ = "Mihai Matei"
__license__ = "BSD"
__email__ = "mihai.matei@my.fmi.unibuc.ro"

import numpy as np
from .data import DataSet

from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


class ModelDataSet:
    def __init__(self, data_set : DataSet):
        self._data_set = data_set
        self._train_features = None
        self._train_target = None
        self._validation_features = None
        self._validation_target = None
        self._test_features = None
        self._test_target = None
        self._feature_shape = None

    def flatten(self):
        self._feature_shape = self.train_features.shape[1:]
        self._train_features = self.train_features.reshape((self.train_features.shape[0], -1))
        self._validation_features = self.validation_features.reshape((self.validation_features.shape[0], -1))
        self._test_features = self.test_features.reshape((self.test_features.shape[0], -1))

        return self

    def un_flatten(self):
        self._train_features = self.train_features.reshape(
            (self.train_features.shape[0], *self._feature_shape))
        self._validation_features = self.validation_features.reshape(
            (self.validation_features.shape[0], *self._feature_shape))
        self._test_features = self.test_features.reshape(
            (self.test_features.shape[0], *self._feature_shape))

        return self

    def augment_train(self, train_features, train_target):
        self._train_features = np.concatenate((self.train_features, train_features), axis=0)
        self._train_target = np.concatenate((self.train_target, train_target), axis=0)

        return self

    def normalize(self, mean=0, std=1):
        norm_transform = StandardScaler()

        # Fit the scale transformer on the training data
        norm_transform.mean_ = np.full((self.train_features.shape[1],), mean)
        norm_transform.var_ = np.full((self.train_features.shape[1],), std**2)

        self._train_features = norm_transform.fit_transform(self.train_features)
        self._validation_features = norm_transform.transform(self.validation_features)
        self._test_features = norm_transform.transform(self.test_features)

        return self

    def normalize_MobileNetV2(self):
        '''
        Special normalization for MobileNet2
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_example.ipynb
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py
        https://arxiv.org/abs/1801.04381
        '''
        self._train_features = self.train_features / 128. - 1.
        self._validation_features = self.validation_features / 128. - 1.
        self._test_features = self.test_features / 128. - 1.

        return self

    def filter(self, max_features=100):
        assert max_features is None or max_features <= self.train_features.shape[1]

        constant_filter = VarianceThreshold(threshold=0.1)
        self._train_features = constant_filter.fit_transform(self.train_features)
        self._validation_features = constant_filter.transform(self.validation_features)
        self._test_features = constant_filter.transform(self.test_features)

        anova_filter = SelectKBest(f_classif, k=max_features)
        self._train_features = anova_filter.fit_transform(self.train_features, self.train_target)
        self._validation_features = anova_filter.transform(self.validation_features)
        self._test_features = anova_filter.transform(self.test_features)

        return self

    @property
    def train_features(self):
        if self._train_features is None:
            self._train_features = self._data_set.train_features.to_numpy()
            self._train_features = np.stack(self._train_features, axis=0)
        return self._train_features

    @property
    def train_target(self):
        if self._train_target is None:
            self._train_target = self._data_set.train_target.cat.codes.to_numpy()
        return self._train_target

    @property
    def validation_features(self):
        if self._validation_features is None:
            self._validation_features = self._data_set.validation_features.to_numpy()
            self._validation_features = np.stack(self._validation_features, axis=0)
        return self._validation_features

    @property
    def validation_target(self):
        if self._validation_target is None:
            self._validation_target = self._data_set.validation_target.cat.codes.to_numpy()
        return self._validation_target

    @property
    def test_features(self):
        if self._test_features is None:
            self._test_features = self._data_set.test_features.to_numpy()
            self._test_features = np.stack(self._test_features, axis=0)
        return self._test_features

    @property
    def test_target(self):
        if self._test_target is None:
            self._test_target = self._data_set.test_target.cat.codes.to_numpy()
        return self._test_target

    @property
    def classes(self):
        return self._data_set.class_ids
