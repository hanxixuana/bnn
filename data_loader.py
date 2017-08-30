#!/usr/bin/env python

from warnings import warn

import numpy as np
import torch


class DataLoader:
    def __init__(self, data, target, batch_size, nn_input_shape, e_exp=None, weight=None, train_param=None):
        if not (isinstance(data, np.ndarray) and isinstance(target, np.ndarray)):
            raise ValueError('Error: data or target is not numpy.ndarray.')
        if not (data.ndim == 4 and (target.ndim == 2 or target.ndim == 4)):
            raise ValueError('Error: data should be four-dimensional and target should be two-dimensional.')
        if data.shape[0] != target.shape[0]:
            raise ValueError('Error: The numbers of samples in data and target should be the same.')

        if batch_size > data.shape[0]:
            warn('batch_size is larger than the sample size, so DataLoader forces batch_size to be the sample size.')
            batch_size = data.shape[0]

        if not (data.dtype == 'float32' and target.dtype == 'float32'):
            warn('The data type of data and target should be float 32. DataLoader has converted them to float32.')
            self.data = data.astype('float32')
            self.target = target.astype('float32')
        else:
            self.data = data
            self.target = target

        if e_exp is None:
            self.e_exp = np.ones([self.data.shape[0], 1], dtype='float32')
        else:
            if not isinstance(e_exp, np.ndarray):
                raise ValueError('Error: e_exp is not numpy.ndarray.')
            if not e_exp.ndim == 2:
                raise ValueError('Error: e_exp should be a two-dimensional column vector.')
            if e_exp.dtype == 'float32':
                self.e_exp = e_exp
            else:
                warn('The data type of data and target should be float 32. DataLoader has converted them to float32.')
                self.e_exp = e_exp.astype('float32')

        if weight is None:
            self.weight = np.ones([self.data.shape[0], 1], dtype='float32')
        else:
            if not isinstance(weight, np.ndarray):
                raise ValueError('Error: weight is not numpy.ndarray.')
            if not weight.ndim == 2:
                raise ValueError('Error: weight should be a two-dimensional column vector.')
            if weight.dtype == 'float32':
                self.weight = weight
            else:
                warn('The data type of data and target should be float 32. DataLoader has converted them to float32.')
                self.weight = weight.astype('float32')

        self.batch_size = int(batch_size)

        self.original_sample_n = data.shape[0]
        self.original_sample_index_list = range(self.original_sample_n)

        self.total_batch_n = self.original_sample_n / self.batch_size
        self.total_sample_n = self.total_batch_n * self.batch_size
        self.total_sample_index_list = self._reset_total_sample_index_list(shuffle=False)

        self.current_batch_idx = -1
        self.sample_idx_in_current_batch_list = []

        self.used_sample_index_list = []

        self.margin = np.zeros([self.original_sample_n] + nn_input_shape, dtype='float32')

    def update_margin(self, margin_update, epoch_idx=None, link=None, bnn=None):
        """
        Update the margin.
        :param torch.FloatTensor margin_update:   [self.data.shape[0]] + nn_input_shape
        :param link: link function
        """
        self.margin += margin_update.cpu().numpy()

    def next_batch(self):
        """
        Get a batch.
        :return:    a boolean value whether the current batch is the last one
        """
        if self.current_batch_idx < self.total_batch_n - 1:
            self.current_batch_idx += 1
            self.sample_idx_in_current_batch_list = self.total_sample_index_list[
                                                    (self.current_batch_idx * self.batch_size):
                                                    (self.current_batch_idx + 1) * self.batch_size
                                                    ]
            self.used_sample_index_list.extend(self.sample_idx_in_current_batch_list)
            return True
        else:
            self.sample_idx_in_current_batch_list = []
            return False

    def get_data_for_nn(self, enable_cuda):
        if enable_cuda:
            return torch.from_numpy(self.margin[self.sample_idx_in_current_batch_list]).cuda(), \
                   torch.from_numpy(self.target[self.sample_idx_in_current_batch_list]).cuda(), \
                   torch.from_numpy(self.e_exp[self.sample_idx_in_current_batch_list]).cuda(), \
                   torch.from_numpy(self.weight[self.sample_idx_in_current_batch_list]).cuda(),
        else:
            return torch.from_numpy(self.margin[self.sample_idx_in_current_batch_list]), \
                   torch.from_numpy(self.target[self.sample_idx_in_current_batch_list]), \
                   torch.from_numpy(self.e_exp[self.sample_idx_in_current_batch_list]), \
                   torch.from_numpy(self.weight[self.sample_idx_in_current_batch_list]),

    def get_data_for_booster_layer(self, enable_cuda):
        if enable_cuda:
            return torch.from_numpy(self.data[self.used_sample_index_list]).cuda(), \
                   torch.from_numpy(self.margin[self.used_sample_index_list]).cuda(), \
                   torch.from_numpy(self.target[self.used_sample_index_list]).cuda(), \
                   torch.from_numpy(self.e_exp[self.used_sample_index_list]).cuda(), \
                   torch.from_numpy(self.weight[self.used_sample_index_list]).cuda()
        else:
            return torch.from_numpy(self.data[self.used_sample_index_list]), \
                   torch.from_numpy(self.margin[self.used_sample_index_list]), \
                   torch.from_numpy(self.target[self.used_sample_index_list]), \
                   torch.from_numpy(self.e_exp[self.used_sample_index_list]), \
                   torch.from_numpy(self.weight[self.used_sample_index_list])

    def get_length_of_data_for_booster_layer(self):
        return self.used_sample_index_list.__len__()

    def reset_used_data(self):
        self.used_sample_index_list = []

    def start_new_round(self, shuffle=True):
        self.current_batch_idx = -1
        self.total_sample_index_list = self._reset_total_sample_index_list(shuffle=shuffle)
        self.sample_idx_in_current_batch_list = []
        self.used_sample_index_list = []

    def _reset_total_sample_index_list(self, shuffle=True):
        if shuffle:
            return np.random.permutation(self.original_sample_index_list)[:self.total_sample_n].tolist()
        else:
            return self.original_sample_index_list[:self.total_sample_n]


class ProbWeightedDataLoader:
    def __init__(self, data, target, batch_size, nn_input_shape, e_exp=None, weight=None, train_param=None):
        if not (isinstance(data, np.ndarray) and isinstance(target, np.ndarray)):
            raise ValueError('Error: data or target is not numpy.ndarray.')
        if not (data.ndim == 4 and (target.ndim == 2 or target.ndim == 4)):
            raise ValueError('Error: data should be four-dimensional and target should be two-dimensional.')
        if data.shape[0] != target.shape[0]:
            raise ValueError('Error: The numbers of samples in data and target should be the same.')

        if batch_size > data.shape[0]:
            warn('batch_size is larger than the sample size, so DataLoader forces batch_size to be the sample size.')
            batch_size = data.shape[0]

        if not (data.dtype == 'float32' and target.dtype == 'float32'):
            warn('The data type of data and target should be float 32. DataLoader has converted them to float32.')
            self.data = data.astype('float32')
            self.target = target.astype('float32')
        else:
            self.data = data
            self.target = target

        if e_exp is None:
            self.e_exp = np.ones([self.data.shape[0], 1], dtype='float32')
        else:
            if not isinstance(e_exp, np.ndarray):
                raise ValueError('Error: e_exp is not numpy.ndarray.')
            if not e_exp.ndim == 2:
                raise ValueError('Error: e_exp should be a two-dimensional column vector.')
            if e_exp.dtype == 'float32':
                self.e_exp = e_exp
            else:
                warn('The data type of data and target should be float 32. DataLoader has converted them to float32.')
                self.e_exp = e_exp.astype('float32')

        if weight is None:
            self.weight = np.ones([self.data.shape[0], 1], dtype='float32')
        else:
            if not isinstance(weight, np.ndarray):
                raise ValueError('Error: weight is not numpy.ndarray.')
            if not weight.ndim == 2:
                raise ValueError('Error: weight should be a two-dimensional column vector.')
            if weight.dtype == 'float32':
                self.weight = weight
            else:
                warn('The data type of data and target should be float 32. DataLoader has converted them to float32.')
                self.weight = weight.astype('float32')

        self.batch_size = int(batch_size)

        self.original_sample_n = data.shape[0]
        self.original_sample_index_list = range(self.original_sample_n)

        self.total_batch_n = self.original_sample_n / self.batch_size
        self.total_sample_n = self.total_batch_n * self.batch_size
        self.total_sample_index_list = self._reset_total_sample_index_list(shuffle=False)

        self.current_batch_idx = -1
        self.sample_idx_in_current_batch_list = []

        self.used_sample_index_list = []

        self.margin = np.zeros([self.original_sample_n] + nn_input_shape, dtype='float32')

        # weighting related code

        if 'use_prob_as_weight_after_n_epoch' in train_param:
            self.use_prob_as_weight_after_n_epoch = train_param['use_prob_as_weight_after_n_epoch']
        else:
            self.use_prob_as_weight_after_n_epoch = 0

    def update_margin(self, margin_update, epoch_idx, link, bnn):
        """
        Update the margin.
        :param torch.FloatTensor margin_update: [self.data.shape[0]] + nn_input_shape
        :param int epoch_idx:                   the index of the current epoch
        :param link:                            link function
        :param bnn:                             bnn instance
        """
        self.margin += margin_update.cpu().numpy()
        if epoch_idx > self.use_prob_as_weight_after_n_epoch:
            self.weight = link(
                bnn.nn_forward(
                    torch.from_numpy(self.margin),
                    requires_grad=False
                ).data,
                torch.from_numpy(self.e_exp)
            ).numpy()

    def next_batch(self):
        """
        Get a batch.
        :return:    a boolean value whether the current batch is the last one
        """
        if self.current_batch_idx < self.total_batch_n - 1:
            self.current_batch_idx += 1
            self.sample_idx_in_current_batch_list = self.total_sample_index_list[
                                                    (self.current_batch_idx * self.batch_size):
                                                    (self.current_batch_idx + 1) * self.batch_size
                                                    ]
            self.used_sample_index_list.extend(self.sample_idx_in_current_batch_list)
            return True
        else:
            self.sample_idx_in_current_batch_list = []
            return False

    def get_data_for_nn(self, enable_cuda):
        if enable_cuda:
            return torch.from_numpy(self.margin[self.sample_idx_in_current_batch_list]).cuda(), \
                   torch.from_numpy(self.target[self.sample_idx_in_current_batch_list]).cuda(), \
                   torch.from_numpy(self.e_exp[self.sample_idx_in_current_batch_list]).cuda(), \
                   torch.from_numpy(self.weight[self.sample_idx_in_current_batch_list]).cuda(),
        else:
            return torch.from_numpy(self.margin[self.sample_idx_in_current_batch_list]), \
                   torch.from_numpy(self.target[self.sample_idx_in_current_batch_list]), \
                   torch.from_numpy(self.e_exp[self.sample_idx_in_current_batch_list]), \
                   torch.from_numpy(self.weight[self.sample_idx_in_current_batch_list]),

    def get_data_for_booster_layer(self, enable_cuda):
        if enable_cuda:
            return torch.from_numpy(self.data[self.used_sample_index_list]).cuda(), \
                   torch.from_numpy(self.margin[self.used_sample_index_list]).cuda(), \
                   torch.from_numpy(self.target[self.used_sample_index_list]).cuda(), \
                   torch.from_numpy(self.e_exp[self.used_sample_index_list]).cuda(), \
                   torch.from_numpy(self.weight[self.used_sample_index_list]).cuda()
        else:
            return torch.from_numpy(self.data[self.used_sample_index_list]), \
                   torch.from_numpy(self.margin[self.used_sample_index_list]), \
                   torch.from_numpy(self.target[self.used_sample_index_list]), \
                   torch.from_numpy(self.e_exp[self.used_sample_index_list]), \
                   torch.from_numpy(self.weight[self.used_sample_index_list])

    def get_length_of_data_for_booster_layer(self):
        return self.used_sample_index_list.__len__()

    def reset_used_data(self):
        self.used_sample_index_list = []

    def start_new_round(self, shuffle=True):
        self.current_batch_idx = -1
        self.total_sample_index_list = self._reset_total_sample_index_list(shuffle=shuffle)
        self.sample_idx_in_current_batch_list = []
        self.used_sample_index_list = []

    def _reset_total_sample_index_list(self, shuffle=True):
        if shuffle:
            return np.random.permutation(self.original_sample_index_list)[:self.total_sample_n].tolist()
        else:
            return self.original_sample_index_list[:self.total_sample_n]


class IncWeightedDataLoader:
    def __init__(self, data, target, batch_size, nn_input_shape, e_exp=None, weight=None, train_param=None):
        if not (isinstance(data, np.ndarray) and isinstance(target, np.ndarray)):
            raise ValueError('Error: data or target is not numpy.ndarray.')
        if not (data.ndim == 4 and (target.ndim == 2 or target.ndim == 4)):
            raise ValueError('Error: data should be four-dimensional and target should be two-dimensional.')
        if data.shape[0] != target.shape[0]:
            raise ValueError('Error: The numbers of samples in data and target should be the same.')

        if batch_size > data.shape[0]:
            warn('batch_size is larger than the sample size, so DataLoader forces batch_size to be the sample size.')
            batch_size = data.shape[0]

        if not (data.dtype == 'float32' and target.dtype == 'float32'):
            warn('The data type of data and target should be float 32. DataLoader has converted them to float32.')
            self.data = data.astype('float32')
            self.target = target.astype('float32')
        else:
            self.data = data
            self.target = target

        if e_exp is None:
            self.e_exp = np.ones([self.data.shape[0], 1], dtype='float32')
        else:
            if not isinstance(e_exp, np.ndarray):
                raise ValueError('Error: e_exp is not numpy.ndarray.')
            if not e_exp.ndim == 2:
                raise ValueError('Error: e_exp should be a two-dimensional column vector.')
            if e_exp.dtype == 'float32':
                self.e_exp = e_exp
            else:
                warn('The data type of data and target should be float 32. DataLoader has converted them to float32.')
                self.e_exp = e_exp.astype('float32')

        if weight is None:
            self.weight = np.ones([self.data.shape[0], 1], dtype='float32')
        else:
            if not isinstance(weight, np.ndarray):
                raise ValueError('Error: weight is not numpy.ndarray.')
            if not weight.ndim == 2:
                raise ValueError('Error: weight should be a two-dimensional column vector.')
            if weight.dtype == 'float32':
                self.weight = weight
            else:
                warn('The data type of data and target should be float 32. DataLoader has converted them to float32.')
                self.weight = weight.astype('float32')

        self.batch_size = int(batch_size)

        self.original_sample_n = data.shape[0]
        self.original_sample_index_list = range(self.original_sample_n)

        self.total_batch_n = self.original_sample_n / self.batch_size
        self.total_sample_n = self.total_batch_n * self.batch_size
        self.total_sample_index_list = self._reset_total_sample_index_list(shuffle=False)

        self.current_batch_idx = -1
        self.sample_idx_in_current_batch_list = []

        self.used_sample_index_list = []

        self.margin = np.zeros([self.original_sample_n] + nn_input_shape, dtype='float32')

        # weight related code

        if 'inc_weight_after_n_epoch' in train_param:
            self.inc_weight_after_n_epoch = train_param['inc_weight_after_n_epoch']
        else:
            self.inc_weight_after_n_epoch = 0
        if 'inc_coef' in train_param:
            self.inc_coef = train_param['inc_coef']
        else:
            self.inc_coef = 1e-5

    def update_margin(self, margin_update, epoch_idx, link=None, bnn=None):
        """
        Update the margin.
        :param torch.FloatTensor margin_update: [self.data.shape[0]] + nn_input_shape
        :param int epoch_idx:                   the index of the current epoch
        :param link:                            link function
        :param bnn:                             bnn instance
        """
        self.margin += margin_update.cpu().numpy()
        if epoch_idx > self.inc_weight_after_n_epoch:
            self.weight += self.target * self.inc_coef

    def next_batch(self):
        """
        Get a batch.
        :return:    a boolean value whether the current batch is the last one
        """
        if self.current_batch_idx < self.total_batch_n - 1:
            self.current_batch_idx += 1
            self.sample_idx_in_current_batch_list = self.total_sample_index_list[
                                                    (self.current_batch_idx * self.batch_size):
                                                    (self.current_batch_idx + 1) * self.batch_size
                                                    ]
            self.used_sample_index_list.extend(self.sample_idx_in_current_batch_list)
            return True
        else:
            self.sample_idx_in_current_batch_list = []
            return False

    def get_data_for_nn(self, enable_cuda):
        if enable_cuda:
            return torch.from_numpy(self.margin[self.sample_idx_in_current_batch_list]).cuda(), \
                   torch.from_numpy(self.target[self.sample_idx_in_current_batch_list]).cuda(), \
                   torch.from_numpy(self.e_exp[self.sample_idx_in_current_batch_list]).cuda(), \
                   torch.from_numpy(self.weight[self.sample_idx_in_current_batch_list]).cuda(),
        else:
            return torch.from_numpy(self.margin[self.sample_idx_in_current_batch_list]), \
                   torch.from_numpy(self.target[self.sample_idx_in_current_batch_list]), \
                   torch.from_numpy(self.e_exp[self.sample_idx_in_current_batch_list]), \
                   torch.from_numpy(self.weight[self.sample_idx_in_current_batch_list]),

    def get_data_for_booster_layer(self, enable_cuda):
        if enable_cuda:
            return torch.from_numpy(self.data[self.used_sample_index_list]).cuda(), \
                   torch.from_numpy(self.margin[self.used_sample_index_list]).cuda(), \
                   torch.from_numpy(self.target[self.used_sample_index_list]).cuda(), \
                   torch.from_numpy(self.e_exp[self.used_sample_index_list]).cuda(), \
                   torch.from_numpy(self.weight[self.used_sample_index_list]).cuda()
        else:
            return torch.from_numpy(self.data[self.used_sample_index_list]), \
                   torch.from_numpy(self.margin[self.used_sample_index_list]), \
                   torch.from_numpy(self.target[self.used_sample_index_list]), \
                   torch.from_numpy(self.e_exp[self.used_sample_index_list]), \
                   torch.from_numpy(self.weight[self.used_sample_index_list])

    def get_length_of_data_for_booster_layer(self):
        return self.used_sample_index_list.__len__()

    def reset_used_data(self):
        self.used_sample_index_list = []

    def start_new_round(self, shuffle=True):
        self.current_batch_idx = -1
        self.total_sample_index_list = self._reset_total_sample_index_list(shuffle=shuffle)
        self.sample_idx_in_current_batch_list = []
        self.used_sample_index_list = []

    def _reset_total_sample_index_list(self, shuffle=True):
        if shuffle:
            return np.random.permutation(self.original_sample_index_list)[:self.total_sample_n].tolist()
        else:
            return self.original_sample_index_list[:self.total_sample_n]
