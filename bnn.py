#!/usr/bin/env python

from copy import deepcopy
from os import makedirs, listdir, getcwd
from pickle import dump, load
from time import time
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import xgboost as xgb
from torch.autograd import Variable
from tqdm import trange

from func import bernoulli_lift_level
from log_to_file import Logger


class BoostNN:
    def __init__(self, **kwargs):
        """
        Connect a neural network with a XGBoost.

        To load a model:
        :param path_prefix:         path + name_prefix
        :param index:               index at the end

        To make a new model
        :param nn_model:            nn
        :param nn_optimizer:        an initialized optimizer of nn
        :param loss:                an initialized loss of nn
        :param feature_dim_list:    a list of [n_channel_to_boosting, first_dim_to_boosting, second_dim_to_boosting]
        :param xgb_param:          a dictionary of parameters for XGB
        :param enable_cuda:         a boolean value

        """

        self.booster_layer_list = None
        self.booster_layer_count_list = None
        self.booster_layer_coef_list = None
        self.booster_layer_seed_list = None

        self.xgb_param = None
        self.nn_param = None
        self.enable_cuda = None

        self.nn_model = None
        self.nn_optimizer = None
        self.nn_loss = None

        self.n_channel_to_boosting = None
        self.first_dim_to_boosting = None
        self.second_dim_to_boosting = None

        self.n_channel_from_boosting = None
        self.first_dim_from_boosting = None
        self.second_dim_from_boosting = None

        self.input_to_nn_model = None
        self.grad_of_input_to_nn_model = None
        self.newly_updated_booster_layer_list = None

        if 'path_prefix' in kwargs:
            path_prefix = kwargs['path_prefix']
            if 'index' in kwargs:
                self.load(path_prefix, kwargs['index'])
            else:
                self.load(path_prefix)
        else:
            self.nn_model = kwargs['nn_model']
            self.nn_model.float()

            self.nn_optimizer = optim.SGD(
                self.nn_model.parameters(),
                lr=kwargs['nn_param']['nn_lr']
            )

            self.nn_loss = kwargs['nn_loss']
            self.nn_loss.float()

            # input feature size of boosting layer is the product of feature_dim_list
            self.n_channel_to_boosting = kwargs['feature_dim_list'][0]
            self.first_dim_to_boosting = kwargs['feature_dim_list'][1]
            self.second_dim_to_boosting = kwargs['feature_dim_list'][2]

            # total No. of GBM objects is the product of the following three
            self.n_channel_from_boosting = self.nn_model.n_channel_from_boosting
            self.first_dim_from_boosting = self.nn_model.first_dim_from_boosting
            self.second_dim_from_boosting = self.nn_model.second_dim_from_boosting

            self._init_param(
                kwargs['xgb_param'],
                kwargs['nn_param']
            )
            self._init_boosting_layer()

    def _init_param(self, xgb_param, nn_param):

        self.enable_cuda = nn_param['enable_cuda']

        xgb_param['silent'] = 0
        xgb_param['num_feature'] = \
            self.n_channel_to_boosting * \
            self.first_dim_to_boosting * \
            self.second_dim_to_boosting
        xgb_param['objective'] = 'reg:linear'
        xgb_param['max_delta_step'] = 0.0

        if self.enable_cuda:
            xgb_param['updater'] = 'grow_gpu'
            self.nn_model.cuda()
        else:
            xgb_param['updater'] = 'grow_colmaker,prune'
            self.nn_model.cpu()

        self.xgb_param = xgb_param.copy()
        self.nn_param = nn_param.copy()

    def _init_boosting_layer(self):

        self.booster_layer_list = [
            [
                [
                    [] for _ in range(self.second_dim_from_boosting)
                    ] for _ in range(self.first_dim_from_boosting)
                ] for _ in range(self.n_channel_from_boosting)
            ]

        self.booster_layer_count_list = [
            [
                [
                    0 for _ in range(self.second_dim_from_boosting)
                    ] for _ in range(self.first_dim_from_boosting)
                ] for _ in range(self.n_channel_from_boosting)
            ]

        self.booster_layer_coef_list = [
            [
                [
                    [] for _ in range(self.second_dim_from_boosting)
                    ] for _ in range(self.first_dim_from_boosting)
                ] for _ in range(self.n_channel_from_boosting)
            ]

        self.booster_layer_seed_list = [
            [
                [
                    int(np.random.rand() * 1e5)
                    for _ in range(self.second_dim_from_boosting)
                    ] for _ in range(self.first_dim_from_boosting)
                ] for _ in range(self.n_channel_from_boosting)
            ]

        self.newly_updated_booster_layer_list = [
            [
                [
                    False for _ in range(self.second_dim_from_boosting)
                    ] for _ in range(self.first_dim_from_boosting)
                ] for _ in range(self.n_channel_from_boosting)
            ]

    def _add_booster_layer(self, channel_idx, first_dim_idx, second_dim_idx):

        self.xgb_param['seed'] = \
            self.booster_layer_seed_list[channel_idx][first_dim_idx][second_dim_idx]

        self.booster_layer_list[channel_idx][first_dim_idx][second_dim_idx].append(
            xgb.Booster(params=self.xgb_param)
        )

        self.booster_layer_count_list[channel_idx][first_dim_idx][second_dim_idx] += 1

        self.newly_updated_booster_layer_list[channel_idx][first_dim_idx][second_dim_idx] = True

    def train_booster_layer(self, dmatrix, grad, hess):
        """
        Train a booster layer.
        :param dmatrix: a dmatrix of
                        [
                            batch_size * add_booster_layer_after_n_batch,
                            n_channel_to_boosting * first_dim_to_boosting * second_dim_to_boosting
                        ]
        :param grad:    a numpy array of
                        [
                            batch_size * add_booster_layer_after_n_batch,
                            n_channel_from_boosting,
                            first_dim_from_boosting,
                            second_dim_from_boosting
                        ]
        :param hess:    a numpy array of
                        [
                            batch_size * add_booster_layer_after_n_batch,
                            n_channel_from_boosting,
                            first_dim_from_boosting,
                            second_dim_from_boosting
                        ]
        """
        dmatrix.set_base_margin(
            np.zeros(
                dmatrix.num_row(),
                dtype='float32'
            )
        )
        for channel_idx in range(self.n_channel_from_boosting):
            for first_dim_idx in range(self.first_dim_from_boosting):
                for second_dim_idx in range(self.second_dim_from_boosting):

                    abs_sum_of_grad = np.mean(
                        np.abs(
                            grad[:, channel_idx, first_dim_idx, second_dim_idx]
                        )
                    )

                    if abs_sum_of_grad > 0.0:
                        # add a booster layer
                        self._add_booster_layer(channel_idx, first_dim_idx, second_dim_idx)

                        scaling_coef = \
                            abs_sum_of_grad \
                            / np.mean(
                                np.abs(
                                    hess[:, channel_idx, first_dim_idx, second_dim_idx]
                                )
                            )

                        scaling_coef = 1.0 if scaling_coef > 1.0 else scaling_coef

                        self.booster_layer_coef_list[channel_idx][first_dim_idx][second_dim_idx].append(
                            scaling_coef
                        )

                        train_grad = \
                            grad[:, channel_idx, first_dim_idx, second_dim_idx] / scaling_coef

                        train_hess = hess[:, channel_idx, first_dim_idx, second_dim_idx] + 1e-8

                        self.booster_layer_list[channel_idx][first_dim_idx][second_dim_idx][-1].boost(
                            dmatrix,
                            train_grad.astype('float32').tolist(),
                            train_hess.astype('float32').tolist()
                        )

    def _grad_hook(self):
        def hook(grad):
            self.grad_of_input_to_nn_model = grad

        return hook

    def nn_forward(self, x, requires_grad=True):
        """
        Forward the net.
        :param x:               a numpy array of
                                [
                                    batch_size,
                                    n_channel_from_boosting,
                                    first_dim_from_boosting,
                                    second_dim_from_boosting
                                ]
        :param requires_grad:   requires gradient or not
        :return: a Variable
        """
        if self.enable_cuda:
            self.input_to_nn_model = \
                Variable(torch.from_numpy(x).cuda().contiguous(), requires_grad=requires_grad)
        else:
            self.input_to_nn_model = \
                Variable(torch.from_numpy(x), requires_grad=requires_grad)

        if requires_grad:
            self.input_to_nn_model.register_hook(self._grad_hook())

        outputs = self.nn_model(self.input_to_nn_model)

        return outputs

    def predict(self, x):
        """
        Predict.
        :param x:   a numpy array of
                    [
                        batch_size,
                        n_channel_to_boosting,
                        first_dim_to_boosting,
                        second_dim_to_boosting
                    ]
        :return: a numpy array of the shape of the output of the network
        """
        dmatrix = xgb.DMatrix(x.reshape(x.shape[0], -1))
        dmatrix.set_base_margin(
            np.zeros(
                dmatrix.num_row(),
                dtype='float32'
            )
        )

        xgb_predictions = np.zeros(
            [
                dmatrix.num_row(),
                self.n_channel_from_boosting,
                self.first_dim_from_boosting,
                self.second_dim_from_boosting
            ],
            dtype='float32'
        )

        for channel_idx in range(self.n_channel_from_boosting):
            for first_dim_idx in range(self.first_dim_from_boosting):
                for second_dim_idx in trange(self.second_dim_from_boosting):
                    xgb_predictions[:, channel_idx, first_dim_idx, second_dim_idx] = \
                        self._booster_predict(
                            dmatrix,
                            channel_idx,
                            first_dim_idx,
                            second_dim_idx
                        )

        predictions = self.nn_forward(xgb_predictions, requires_grad=False)

        if self.enable_cuda:
            return predictions.data.cpu().numpy()
        else:
            return predictions.data.numpy()

    def _booster_predict(self, dmatrix, channel_idx, first_dim_idx, second_dim_idx):

        predictions = np.zeros(dmatrix.num_row(), dtype='float32')

        for booster_layer_idx in range(self.booster_layer_count_list[channel_idx][first_dim_idx][second_dim_idx]):
            predictions += \
                self.booster_layer_list[channel_idx][first_dim_idx][second_dim_idx][booster_layer_idx] \
                    .predict(dmatrix) \
                * self.booster_layer_coef_list[channel_idx][first_dim_idx][second_dim_idx][booster_layer_idx]

        return predictions

    def predict_with_newly_added_booster(self, dmatrix, reset_newly_updated_booster_layer_list):

        predictions = np.zeros(
            [
                dmatrix.num_row(),
                self.n_channel_from_boosting,
                self.first_dim_from_boosting,
                self.second_dim_from_boosting
            ]
        )

        dmatrix.set_base_margin(
            np.zeros(
                dmatrix.num_row(),
                dtype='float32'
            )
        )

        for channel_idx in range(self.n_channel_from_boosting):
            for first_dim_idx in range(self.first_dim_from_boosting):
                for second_dim_idx in range(self.second_dim_from_boosting):

                    if self.newly_updated_booster_layer_list[channel_idx][first_dim_idx][second_dim_idx]:

                        predictions[:, channel_idx, first_dim_idx, second_dim_idx] = \
                            self.booster_layer_list[channel_idx][first_dim_idx][second_dim_idx][-1] \
                                .predict(dmatrix)

                        predictions[:, channel_idx, first_dim_idx, second_dim_idx] *= \
                            self.booster_layer_coef_list[channel_idx][first_dim_idx][second_dim_idx][-1]
                        if reset_newly_updated_booster_layer_list:
                            self.newly_updated_booster_layer_list[channel_idx][first_dim_idx][second_dim_idx] = False

        return predictions

    def save(self, path, index=None):
        if index:
            file_name = '%s/%s/bnn_%d.dat' % (getcwd(), path, index)
        else:
            file_name = '%s/%s/bnn.dat' % (getcwd(), path)
        if path not in listdir(getcwd()):
            makedirs(path)
        with open(file_name, 'wb') as fp:
            dump(
                {
                    'booster_layer_list': self.booster_layer_list,
                    'booster_layer_count_list': self.booster_layer_count_list,
                    'booster_layer_coef_list': self.booster_layer_coef_list,
                    'booster_layer_seed_list': self.booster_layer_seed_list,

                    'xgb_param': self.xgb_param,
                    'nn_param': self.nn_param,

                    'nn_model': self.nn_model,
                    'nn_optimizer': self.nn_optimizer,
                    'nn_loss': self.nn_loss,

                    'dims_to_booster_layer': [
                        self.n_channel_to_boosting,
                        self.first_dim_to_boosting,
                        self.second_dim_to_boosting
                    ],
                    'dims_from_booster_layer': [
                        self.n_channel_from_boosting,
                        self.first_dim_from_boosting,
                        self.second_dim_from_boosting
                    ]
                },
                fp
            )

    def load(self, path, index=None):
        if index:
            file_name = '%s/bnn_%d.dat' % (path, index)
        else:
            file_name = '%s/bnn.dat' % path
        with open(file_name, 'rb') as fp:
            loaded_content = load(fp)

            self.booster_layer_list = loaded_content['booster_layer_list']
            self.booster_layer_count_list = loaded_content['booster_layer_count_list']
            self.booster_layer_coef_list = loaded_content['booster_layer_coef_list']
            self.booster_layer_seed_list = loaded_content['booster_layer_seed_list']

            self.xgb_param = loaded_content['xgb_param']
            self.nn_param = loaded_content['nn_param']

            self.nn_model = loaded_content['nn_model']
            self.nn_optimizer = loaded_content['nn_optimizer']
            self.nn_loss = loaded_content['nn_loss']

            self.n_channel_to_boosting = loaded_content['dims_to_booster_layer'][0]
            self.first_dim_to_boosting = loaded_content['dims_to_booster_layer'][1]
            self.second_dim_to_boosting = loaded_content['dims_to_booster_layer'][2]

            self.n_channel_from_boosting = loaded_content['dims_from_booster_layer'][0]
            self.first_dim_from_boosting = loaded_content['dims_from_booster_layer'][1]
            self.second_dim_from_boosting = loaded_content['dims_from_booster_layer'][2]

        self.enable_cuda = self.nn_param['enable_cuda']


class DataLoader:
    def __init__(self, data, target, batch_size, nn_input_shape):
        if not (isinstance(data, np.ndarray) and isinstance(target, np.ndarray)):
            raise ValueError('Error: data or target is not numpy.ndarray.')

        if not (data.ndim == 4 and (target.ndim == 2 or target.ndim == 4)):
            raise ValueError('Error: data should be four-dimensional and target should be two-dimensional.')

        if data.shape[0] != target.shape[0]:
            raise ValueError('Error: The numbers of samples in data and target should be the same.')

        if not (data.dtype == 'float32' and target.dtype == 'float32'):
            warn('The data type of data and target should be float 32. DataLoader has converted them to float32.')

        if batch_size > data.shape[0]:
            warn('batch_size is larger than the sample size, so DataLoader forces batch_size to be the sample size.')
            batch_size = data.shape[0]

        self.data = data
        self.target = target
        self.batch_size = int(batch_size)

        self.original_sample_n = data.shape[0]
        self.original_sample_index_list = range(self.original_sample_n)

        self.total_batch_n = data.shape[0] / self.batch_size
        self.total_sample_n = self.total_batch_n * self.batch_size

        self.total_sample_index_list = self._reset_total_sample_index_list(shuffle=False)

        self.current_batch_idx = -1
        self.sample_idx_in_current_batch_list = []

        self.used_sample_index_list = []

        self.margin = np.zeros([self.data.shape[0]] + nn_input_shape).astype('float32')

    def update_margin(self, margin_update):
        """
        Update the margin.
        :param margin_update:   a numpy array of [self.data.shape[0], nn_input_shape]
        """
        self.margin += margin_update

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

    def get_data_for_nn(self):
        return self.margin[self.sample_idx_in_current_batch_list], \
               self.target[self.sample_idx_in_current_batch_list]

    def get_data_for_booster_layer(self):
        return self.data[self.used_sample_index_list], \
               self.margin[self.used_sample_index_list], \
               self.target[self.used_sample_index_list]

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


class Trainer:
    def __init__(self, model, train_param):

        # initialize a logger
        self.logger = Logger('main', to_stdout=train_param['verbose'], path='log')
        self.logger.log('__init__ starts...', 'Trainer.__init__()')
        self.logger.log('NN Name: ' + model.nn_model.__class__.__name__, 'Trainer.__init__()')
        self.logger.log('NN Structure: \n' + model.nn_model.__repr__(), 'Trainer.__init__()')
        self.logger.log('================= Training param =================',
                        'Trainer.__init__()')
        for key in train_param:
            self.logger.log(key + ': ' + str(train_param[key]), 'Trainer.__init__()')
        self.logger.log('================= XGBoost param ==================',
                        'Trainer.__init__()')
        for key in model.xgb_param:
            self.logger.log(key + ': ' + str(model.xgb_param[key] if key is not None else ' '), 'Trainer.__init__()')
        self.logger.log('===================================================',
                        'Trainer.__init__()')

        torch.set_num_threads(train_param['torch_nthread'])

        # some attributes
        self.train_param = deepcopy(train_param)

        self.grad_ma_coef = train_param['adam_grad_and_hess_ma_coef_list'][0]
        self.hess_ma_coef = train_param['adam_grad_and_hess_ma_coef_list'][1]
        self.target_regularization_coef = train_param['target_regularization_coef']

        self.model = model

        self.classification_threshold = 0.5

        # some float tensors
        self.std_per_feature_float_tensor = None
        self.max_per_feature_float_tensor = None
        self.min_per_feature_float_tensor = None

        # for data
        self.train_data = None
        self.train_target = None
        self.val_data = None
        self.val_target = None

        self.train_set_loader = None
        self.val_set_loader = None

        # for adam
        self.ma_grad_array = None
        self.ma_hess_array = None
        self.count_of_booster_layer_training = 0

        # for recording
        self.average_tr_loss_list = None
        self.average_val_loss_list = None

        self.grad_scale_list = None
        self.hess_scale_list = None

        self.val_acc_list = None

        self.count_of_warming_and_training = -1

        self.logger.log('__init__ ends...\n', 'Trainer.__init__()')

    @staticmethod
    def _start_string(phase, output_0, output_1, output_2, output_3, output_4):
        return ' ' + phase + ' %6d, Batch [%3d/%3d], max No. of boosters added: %3d, time: %.4f' \
                             % (output_0, output_1, output_2, output_3, output_4)

    @staticmethod
    def _blank_spaces(phase):
        return ' ' * (' ' + phase + ' %6d, ' % 0).__len__()

    def input_data(self, train_data, train_target, val_data, val_target, make_copy_of_data=False):

        if make_copy_of_data:
            self.train_data = deepcopy(train_data)
            self.train_target = deepcopy(train_target)
            self.val_data = deepcopy(val_data)
            self.val_target = deepcopy(val_target)
        else:
            self.train_data = train_data
            self.train_target = train_target
            self.val_data = val_data
            self.val_target = val_target

        if self.train_param['normalize_to_0_1']:
            self.train_data -= self.train_data[~np.isnan(self.train_data)].min()
            self.train_data /= self.train_data[~np.isnan(self.train_data)].max()
            self.val_data -= self.val_data[~np.isnan(self.val_data)].min()
            self.val_data /= self.val_data[~np.isnan(self.val_data)].max()

        self.logger.log(
            'train_data max: ' + str(self.train_data[~np.isnan(self.train_data)].max()) +
            '\t\ttrain_data min:  ' + str(self.train_data[~np.isnan(self.train_data)].min()) +
            '\t\tval_data max:  ' + str(self.val_data[~np.isnan(self.val_data)].max()) +
            '\t\tval_data min:  ' + str(self.val_data[~np.isnan(self.val_data)].min()),
            'Trainer.__init__()'
        )

        self.std_per_feature_float_tensor = self._per_feature_operation(self.train_data, np.std, self.logger)
        self.max_per_feature_float_tensor = self._per_feature_operation(self.train_data, np.max, self.logger)
        self.min_per_feature_float_tensor = self._per_feature_operation(self.train_data, np.min, self.logger)

        # put data into a DataLoader
        self.train_set_loader = DataLoader(
            train_data,
            train_target,
            self.train_param['batch_size'],
            [
                self.model.n_channel_from_boosting,
                self.model.first_dim_from_boosting,
                self.model.second_dim_from_boosting
            ]
        )

        self.val_set_loader = DataLoader(
            val_data,
            val_target,
            val_data.shape[0],
            [
                self.model.n_channel_from_boosting,
                self.model.first_dim_from_boosting,
                self.model.second_dim_from_boosting
            ]
        )
        self.logger.log('train_set_loader and val_set_loader DONE.\n', 'Trainer.input_data()')

    @staticmethod
    def _per_feature_operation(data, fun, logger=None):
        sample_n = data.shape[0]
        result = np.array(
            [
                float(
                    fun(
                        item[
                            ~np.isnan(item)
                        ]
                    )
                )
                if np.sum(np.isnan(item)) < sample_n else 1.0
                for item in data.reshape(data.shape[0], -1).transpose()
                ]
        )
        if logger:
            range_string = 'feature ' + fun.__name__ + ': \n'
            for idx, item in enumerate(result):
                range_string += ' ' * 10 + 'f_' + str(idx) + ': %.4f\t' % item
                if idx % 5 == 4:
                    range_string += '\n'
            logger.log(range_string, 'Trainer._per_feature_operation()')
        return torch.from_numpy(result)

    def _update_ma(self, grad_long_double_array, hess_long_double_array):

        if self.ma_grad_array is None or self.ma_hess_array is None:
            self.ma_grad_array = (1.0 - self.grad_ma_coef) * grad_long_double_array
            self.ma_hess_array = (1.0 - self.hess_ma_coef) * hess_long_double_array
        else:
            self.ma_grad_array = \
                self.grad_ma_coef * self.ma_grad_array \
                + (1.0 - self.grad_ma_coef) * grad_long_double_array
            self.ma_hess_array = \
                self.hess_ma_coef * self.ma_hess_array \
                + (1.0 - self.hess_ma_coef) * hess_long_double_array

    def _nn_step(self, margin_float_array, target_float_array, mode):

        if self.model.enable_cuda:
            target_float_variable = Variable(torch.from_numpy(target_float_array).cuda())
        else:
            target_float_variable = Variable(torch.from_numpy(target_float_array))

        if mode == 'loss_and_backward_and_update':

            self.model.nn_optimizer.zero_grad()

            output_float_variable = self.model.nn_forward(
                margin_float_array,
                requires_grad=True
            )

            corrected_target_float_variable = self._correct_target(
                output_float_variable,
                target_float_variable
            )

            loss_variable, display_loss_variable = self._cal_loss(
                output_float_variable,
                target_float_variable,
                corrected_target_float_variable
            )

            loss_variable.backward()
            self.model.nn_optimizer.step()

            return loss_variable.data[0], display_loss_variable.data[0]

        elif mode == 'loss_and_backward':

            self.model.nn_optimizer.zero_grad()

            output_float_variable = self.model.nn_forward(
                margin_float_array,
                requires_grad=True
            )

            corrected_target_float_variable = self._correct_target(
                output_float_variable,
                target_float_variable
            )

            loss_variable, display_loss_variable = self._cal_loss(
                output_float_variable,
                target_float_variable,
                corrected_target_float_variable
            )

            loss_variable.backward()

            return loss_variable.data[0], \
                   display_loss_variable.data[0], \
                   output_float_variable, \
                   corrected_target_float_variable

        elif mode == 'loss':

            output_float_variable = self.model.nn_forward(
                margin_float_array,
                requires_grad=False
            )

            display_loss_variable = self._cal_loss(
                output_float_variable,
                target_float_variable,
            )

            if self.model.enable_cuda:
                return display_loss_variable.data[0], output_float_variable.data.cpu().numpy()
            else:
                return display_loss_variable.data[0], output_float_variable.data.numpy()

    def _xgb_step(self, dmatrix, grad_long_double_array, hess_long_double_array):
        """
        Add a booster layer for those non-zero gradients.
        :param dmatrix:                 a dmatrix of
                                        [
                                            batch_size * add_booster_layer_after_n_batch,
                                            n_channel_to_boosting * first_dim_to_boosting * second_dim_to_boosting
                                        ]
        :param grad_long_double_array:  a numpy array of
                                        [
                                            batch_size * add_booster_layer_after_n_batch,
                                            n_channel_from_boosting,
                                            first_dim_from_boosting,
                                            second_dim_from_boosting
                                        ]
        :param hess_long_double_array:    a numpy array of
                                        [
                                            batch_size * add_booster_layer_after_n_batch,
                                            n_channel_from_boosting,
                                            first_dim_from_boosting,
                                            second_dim_from_boosting
                                        ]
        """

        self._update_ma(grad_long_double_array, hess_long_double_array)

        self.model.train_booster_layer(
            dmatrix,
            self.ma_grad_array / (1.0 - self.grad_ma_coef ** (self.count_of_booster_layer_training + 1.0)),
            self.ma_hess_array / (1.0 - self.hess_ma_coef ** (self.count_of_booster_layer_training + 1.0))
        )

        self.count_of_booster_layer_training += 1

    def _correct_target(self, output_float_variable, target_float_variable):
        """
        Correct outputs
        :param output_float_variable:       a torch variable of
                                            [
                                                batch_size,
                                                target.shape[0],
                                                target.shape[1],
                                                ...
                                            ]
                                            or
                                            [
                                                batch_size * add_booster_layer_after_n_batch,
                                                target.shape[0],
                                                target.shape[1],
                                                ...
                                            ]
        :param target_float_variable:       a torch variable of
                                            [
                                                batch_size,
                                                target.shape[0],
                                                target.shape[1],
                                                ...
                                            ]
                                            or
                                            [
                                                batch_size * add_booster_layer_after_n_batch,
                                                target.shape[0],
                                                target.shape[1],
                                                ...
                                            ]
        :return:                            a torch variable of the same shape of output and target
        """

        corrected_target_float_variable = \
            self.target_regularization_coef \
            * target_float_variable \
            + \
            (1.0 - self.target_regularization_coef) \
            * self.model.nn_loss.link(output_float_variable)

        return corrected_target_float_variable

    def _cal_loss(self, output_float_variable, target_float_variable, corrected_target_float_variable=None):

        loss_variable = None

        if self.model.nn_loss.__class__.__name__ == 'BernoulliLoss':

            if corrected_target_float_variable:
                loss_variable = self.model.nn_loss(
                    output_float_variable,
                    corrected_target_float_variable
                )

            display_loss_variable = self.model.nn_loss(
                output_float_variable,
                target_float_variable
            )

        elif self.model.nn_loss.__class__.__name__ == 'FeatureNormalizedMSE':

            if corrected_target_float_variable:
                loss_variable = self.model.nn_loss(
                    output_float_variable.double(),
                    corrected_target_float_variable,
                    1.0 / self.std_per_feature_float_tensor
                )

            display_loss_variable = self.model.nn_loss(
                output_float_variable,
                target_float_variable,
                1.0 / self.std_per_feature_float_tensor
            )

        else:
            raise NotImplementedError

        if corrected_target_float_variable:
            return loss_variable, display_loss_variable
        else:
            return display_loss_variable

    def _cal_grad_and_hess(self, output_float_variable, corrected_target_float_variable):

        if self.model.enable_cuda:
            grad_long_double_array = \
                self.model \
                    .grad_of_input_to_nn_model \
                    .data \
                    .cpu() \
                    .numpy() \
                    .astype('longdouble')
        else:
            grad_long_double_array = \
                self.model \
                    .grad_of_input_to_nn_model \
                    .data \
                    .numpy() \
                    .astype('longdouble')

        if self.model.nn_loss.__class__.__name__ == 'BernoulliLoss':

            hess_long_double_array = self.model.nn_loss.hess(
                grad_long_double_array,
                torch.exp(output_float_variable.data).cpu().numpy().astype('longdouble'),
                corrected_target_float_variable.data.cpu().numpy().astype('longdouble')
            )

        elif self.model.nn_loss.__class__.__name__ == 'FeatureNormalizedMSE':

            hess_long_double_array = self.model.nn_loss.hess(grad_long_double_array)

        else:
            raise NotImplementedError

        return grad_long_double_array, hess_long_double_array

    def _epoch_train(self, epoch):

        epoch_start_time = time()

        running_loss = 0.0
        running_display_loss = 0.0

        self.train_set_loader.start_new_round()

        while self.train_set_loader.next_batch():

            time_start = time()

            margin_float_array, target_float_array = self.train_set_loader.get_data_for_nn()
            loss, display_loss = self._nn_step(
                margin_float_array,
                target_float_array,
                'loss_and_backward_and_update'
            )

            running_loss += loss
            running_display_loss += display_loss

            nn_time = time() - time_start

            self.logger.log(
                self._start_string(
                    'Epoch',
                    epoch + 1,
                    self.train_set_loader.current_batch_idx + 1,
                    self.train_set_loader.total_batch_n,
                    self.count_of_booster_layer_training,
                    nn_time
                ),
                'Trainer._epoch_train()'
            )

            self.logger.log(
                self._blank_spaces(
                    'Epoch'
                )
                + 'corrected batch train loss: %.16lf' %
                (
                    loss / self.train_set_loader.batch_size,
                ),
                'Trainer._epoch_train()'
            )

            self.logger.log(
                self._blank_spaces(
                    'Epoch'
                )
                + 'original batch train loss: %.16lf' %
                (
                    display_loss / self.train_set_loader.batch_size
                ),
                'Trainer._epoch_train()'
            )

            add_booster_layer = \
                self.train_set_loader.current_batch_idx \
                % self.train_param['add_booster_layer_after_n_batch'] \
                == self.train_param['add_booster_layer_after_n_batch'] - 1

            if add_booster_layer:

                data_float_array, _, target_float_array = \
                    self.train_set_loader.get_data_for_booster_layer()

                for booster_layer_idx in range(self.train_param['n_booster_layer_to_add']):
                    time_start = time()

                    _, margin_float_array, _ = \
                        self.train_set_loader.get_data_for_booster_layer()

                    loss, display_loss, output_float_variable, corrected_target_float_variable = self._nn_step(
                        margin_float_array,
                        target_float_array,
                        'loss_and_backward'
                    )

                    grad_long_double_array, hess_long_double_array = self._cal_grad_and_hess(
                        output_float_variable,
                        corrected_target_float_variable
                    )

                    self.grad_scale_list.append(np.mean(np.abs(grad_long_double_array)))
                    self.hess_scale_list.append(np.mean(np.abs(hess_long_double_array)))

                    self._xgb_step(
                        xgb.DMatrix(
                            data_float_array.reshape(
                                data_float_array.shape[0],
                                -1
                            )
                        ),
                        grad_long_double_array,
                        hess_long_double_array
                    )

                    train_margin_update = self.model.predict_with_newly_added_booster(
                        xgb.DMatrix(
                            self.train_set_loader.data.reshape(
                                self.train_set_loader.original_sample_n,
                                -1
                            )
                        ),
                        False
                    )

                    self.train_set_loader.update_margin(train_margin_update)

                    self.val_set_loader.update_margin(
                        self.model.predict_with_newly_added_booster(
                            xgb.DMatrix(
                                self.val_set_loader.data.reshape(
                                    self.val_set_loader.original_sample_n,
                                    -1
                                )
                            ),
                            True
                        )
                    )

                    booster_time = time() - time_start

                    self.logger.log(
                        self._start_string(
                            'Epoch',
                            epoch + 1,
                            self.train_set_loader.current_batch_idx + 1,
                            self.train_set_loader.total_batch_n,
                            self.count_of_booster_layer_training,
                            booster_time
                        ),
                        'Trainer._epoch_train()'
                    )

                    self.logger.log(
                        self._blank_spaces(
                            'Epoch'
                        )
                        + 'L1 norm of gradients: %16.8lf' %
                        (
                            self.grad_scale_list[-1],
                        ),
                        'Trainer._epoch_train()'
                    )

                    self.logger.log(
                        self._blank_spaces(
                            'Epoch'
                        )
                        + 'L1 norm of Hessian: %16.8lf' %
                        (
                            self.hess_scale_list[-1]
                        ),
                        'Trainer._epoch_train()'
                    )

                    self.logger.log(
                        self._blank_spaces(
                            'Epoch',
                        )
                        + 'L1 norm of %d values of margin update: %.16lf' %
                        (
                            float(np.prod(self.train_set_loader.margin.shape)),
                            float(np.mean(np.abs(train_margin_update)))
                        ),
                        'Trainer._epoch_train()'
                    )

                    self.logger.log(
                        self._blank_spaces(
                            'Epoch',
                        )
                        + 'corrected batch train loss: %.16lf' %
                        (
                            loss / self.train_set_loader.get_length_of_data_for_booster_layer(),
                        ),
                        'Trainer._epoch_train()'
                    )

                    self.logger.log(
                        self._blank_spaces(
                            'Epoch',
                        )
                        + 'original batch train loss: %.16lf' %
                        (
                            display_loss / self.train_set_loader.get_length_of_data_for_booster_layer()
                        ),
                        'Trainer._epoch_train()'
                    )

                self.train_set_loader.reset_used_data()

        average_running_loss = running_loss / self.train_set_loader.total_sample_n
        average_running_display_loss = running_display_loss / self.train_set_loader.total_sample_n

        self.logger.log(
            ' Epoch %6d ==>> average train loss: %.16lf' %
            (
                epoch + 1,
                average_running_display_loss
            ),
            'Trainer._epoch_train()'
        )
        self.logger.log(
            '                   average corrected train loss: %.16lf' %
            average_running_loss,
            'Trainer._epoch_train()'
        )
        self.logger.log(
            '                   elapse time: %16.8lf \n\n' %
            (
                time() - epoch_start_time
            ),
            'Trainer._epoch_train()'
        )

        return average_running_display_loss

    def _epoch_val(self, epoch):

        epoch_start_time = time()

        running_display_loss = 0.0

        target_float_array_list = None
        output_float_array_list = None
        if self.model.nn_loss.__class__.__name__ == 'BernoulliLoss':
            target_float_array_list = []
            output_float_array_list = []

        self.val_set_loader.start_new_round(shuffle=False)

        while self.val_set_loader.next_batch():

            margin_float_array, target_float_array = self.val_set_loader.get_data_for_nn()

            display_loss, output_float_array = self._nn_step(
                margin_float_array,
                target_float_array,
                'loss'
            )

            if self.model.nn_loss.__class__.__name__ == 'BernoulliLoss':
                target_float_array_list.append(target_float_array)
                output_float_array_list.append(output_float_array)

            running_display_loss += display_loss

        average_running_display_loss = running_display_loss / self.val_set_loader.total_sample_n

        self.logger.log(
            ' Epoch %6d ==>> average val loss: %.16lf' %
            (
                epoch + 1,
                average_running_display_loss
            ),
            'Trainer._epoch_val()'
        )
        self.logger.log(
            '                   elapse time: %16.8lf' %
            (
                time() - epoch_start_time
            ),
            'Trainer._epoch_val()'
        )

        if self.model.nn_loss.__class__.__name__ == 'BernoulliLoss':

            total_target = np.concatenate(target_float_array_list)
            total_output = np.concatenate(output_float_array_list)

            pct = bernoulli_lift_level(
                total_target[:, 0],
                total_output[:, 0],
                self.train_param['lift_level_at']
            )
            self.logger.log(
                ' Epoch %6d ==>> accuracy achieved for %.4f %% of top predictions: %.4f %% \n\n' %
                (
                    epoch + 1,
                    self.train_param['lift_level_at'] * 100.0,
                    pct * 100.0
                ),
                'Trainer._epoch_val()'
            )

            return average_running_display_loss, pct

        elif self.model.nn_loss.__class__.__name__ == 'FeatureNormalizedMSE':

            return average_running_display_loss

        else:
            raise NotImplementedError

    def _train_loss(self, epoch):

        epoch_start_time = time()

        running_display_loss = 0.0

        self.train_set_loader.start_new_round(shuffle=False)

        while self.train_set_loader.next_batch():
            margin_float_array, target_float_array = self.train_set_loader.get_data_for_nn()

            display_loss, _ = self._nn_step(
                margin_float_array,
                target_float_array,
                'loss'
            )

            running_display_loss += display_loss

        average_running_display_loss = running_display_loss / self.train_set_loader.total_sample_n

        self.logger.log(
            ' Epoch %6d ==>> average train loss: %.16lf' %
            (
                epoch + 1,
                average_running_display_loss
            ),
            'Trainer._train_loss()'
        )
        self.logger.log(
            '                   elapse time: %16.8lf \n\n' %
            (
                time() - epoch_start_time
            ),
            'Trainer._train_loss()'
        )

        return average_running_display_loss

    def _epoch_warming_up(self, epoch):

        epoch_start_time = time()

        running_loss = 0.0
        running_display_loss = 0.0

        self.train_set_loader.start_new_round()

        while self.train_set_loader.next_batch():

            # add_booster_layer = \
            #     self.train_set_loader.current_batch_idx \
            #     % self.train_param['add_booster_layer_after_n_batch'] \
            #     == self.train_param['add_booster_layer_after_n_batch'] - 1 \
            #     or \
            #     self.train_set_loader.current_batch_idx \
            #     == self.train_set_loader.total_batch_n - 1

            add_booster_layer = \
                self.train_set_loader.current_batch_idx \
                % self.train_param['add_booster_layer_after_n_batch'] \
                == self.train_param['add_booster_layer_after_n_batch'] - 1

            if add_booster_layer:
                time_start = time()

                data_float_array, margin_float_array, target_float_array = \
                    self.train_set_loader.get_data_for_booster_layer()

                loss, display_loss, output_float_variable, corrected_target_float_variable = self._nn_step(
                    margin_float_array,
                    target_float_array,
                    'loss_and_backward'
                )

                running_loss += loss
                running_display_loss += display_loss

                grad_long_double_array, hess_long_double_array = self._cal_grad_and_hess(
                    output_float_variable,
                    corrected_target_float_variable
                )

                self.grad_scale_list.append(np.mean(np.abs(grad_long_double_array)))
                self.hess_scale_list.append(np.mean(np.abs(hess_long_double_array)))

                self._xgb_step(
                    xgb.DMatrix(
                        data_float_array.reshape(
                            data_float_array.shape[0],
                            -1
                        )
                    ),
                    grad_long_double_array,
                    hess_long_double_array
                )

                train_margin_update = self.model.predict_with_newly_added_booster(
                    xgb.DMatrix(
                        self.train_set_loader.data.reshape(
                            self.train_set_loader.original_sample_n,
                            -1
                        )
                    ),
                    False
                )

                self.train_set_loader.update_margin(train_margin_update)

                self.val_set_loader.update_margin(
                    self.model.predict_with_newly_added_booster(
                        xgb.DMatrix(
                            self.val_set_loader.data.reshape(
                                self.val_set_loader.original_sample_n,
                                -1
                            )
                        ),
                        True
                    )
                )

                booster_time = time() - time_start

                self.logger.log(
                    self._start_string(
                        'Warming-Up',
                        epoch + 1,
                        self.train_set_loader.current_batch_idx + 1,
                        self.train_set_loader.total_batch_n,
                        self.count_of_booster_layer_training,
                        booster_time
                    ),
                    'Trainer._epoch_warming_up()'
                )

                self.logger.log(
                    self._blank_spaces(
                        'Warming-Up'
                    )
                    + 'L1 norm of gradients: %16.8lf' %
                    (
                        self.grad_scale_list[-1],
                    ),
                    'Trainer._epoch_warming_up()'
                )

                self.logger.log(
                    self._blank_spaces(
                        'Warming-Up'
                    )
                    + 'L1 norm of Hessian: %16.8lf' %
                    (
                        self.hess_scale_list[-1]
                    ),
                    'Trainer._epoch_warming_up()'
                )

                self.logger.log(
                    self._blank_spaces(
                        'Warming-Up',
                    )
                    + 'L1 norm of %d values of margin update: %.16lf' %
                    (
                        float(np.prod(self.train_set_loader.margin.shape)),
                        float(np.mean(np.abs(train_margin_update)))
                    ),
                    'Trainer._epoch_warming_up()'
                )

                self.logger.log(
                    self._blank_spaces(
                        'Warming-Up',
                    )
                    + 'corrected batch train loss: %.16lf' %
                    (
                        loss / self.train_set_loader.get_length_of_data_for_booster_layer(),
                    ),
                    'Trainer._epoch_warming_up()'
                )

                self.logger.log(
                    self._blank_spaces(
                        'Warming-Up',
                    )
                    + 'original batch train loss: %.16lf' %
                    (
                        loss / self.train_set_loader.get_length_of_data_for_booster_layer(),
                    ),
                    'Trainer._epoch_warming_up()'
                )

                self.train_set_loader.reset_used_data()

        average_running_loss = running_loss / self.train_set_loader.total_sample_n
        average_running_display_loss = running_display_loss / self.train_set_loader.total_sample_n

        self.logger.log(
            ' Warming-Up %6d ==>> average train loss: %.16lf' %
            (
                epoch + 1,
                average_running_display_loss
            ),
            'Trainer._epoch_warming_up()'
        )
        self.logger.log(
            '                        average corrected train loss: %.16lf' %
            average_running_loss,
            'Trainer._epoch_warming_up()'
        )
        self.logger.log(
            '                        elapse time: %16.8lf \n\n' %
            (
                time() - epoch_start_time
            ),
            'Trainer._epoch_warming_up()'
        )

        return average_running_display_loss

    def _init_variables_for_recording(self):

        if not self.count_of_booster_layer_training:
            self.count_of_booster_layer_training = 0

        if not self.average_tr_loss_list:
            self.average_tr_loss_list = []
        if not self.average_val_loss_list:
            self.average_val_loss_list = []

        if not self.grad_scale_list:
            self.grad_scale_list = []
        if not self.hess_scale_list:
            self.hess_scale_list = []

        if not self.val_acc_list:
            if self.model.nn_loss.__class__.__name__ == 'BernoulliLoss':
                self.val_acc_list = []
            elif self.model.nn_loss.__class__.__name__ == 'FeatureNormalizedMSE':
                pass
            else:
                raise NotImplementedError

    def warming_up(self, n_epoch_warming_up):

        self.logger.log(
            'Number of warming-up epochs: %d' % n_epoch_warming_up,
            'Trainer.warming_up()'
        )
        self.logger.log(
            'Loss function: %s' % str(self.model.nn_loss)[:-3],
            'Trainer.warming_up()'
        )
        self.logger.log(
            'Warming-Up starts... \n',
            'Trainer.warming_up()'
        )

        self._init_variables_for_recording()

        self.average_tr_loss_list.append(
            self._train_loss(
                self.count_of_warming_and_training
            )
        )

        if self.model.nn_loss.__class__.__name__ == 'BernoulliLoss':
            val_loss, val_acc = self._epoch_val(self.count_of_warming_and_training)
            self.average_val_loss_list.append(val_loss)
            self.val_acc_list.append(val_acc)
        elif self.model.nn_loss.__class__.__name__ == 'FeatureNormalizedMSE':
            self.average_val_loss_list.append(
                self._epoch_val(self.count_of_warming_and_training)
            )
        else:
            raise NotImplementedError

        for warming_up_idx in range(n_epoch_warming_up):

            self.count_of_warming_and_training += 1

            self.average_tr_loss_list.append(
                self._epoch_warming_up(
                    self.count_of_warming_and_training
                )
            )

            if self.model.nn_loss.__class__.__name__ == 'BernoulliLoss':
                val_loss, val_acc = self._epoch_val(self.count_of_warming_and_training)
                self.average_val_loss_list.append(val_loss)
                self.val_acc_list.append(val_acc)
            elif self.model.nn_loss.__class__.__name__ == 'FeatureNormalizedMSE':
                self.average_val_loss_list.append(
                    self._epoch_val(self.count_of_warming_and_training)
                )
            else:
                raise NotImplementedError

        self.logger.log(
            '========Warming-Up Finished========',
            'Trainer.warming_up()'
        )

        if self.model.nn_loss.__class__.__name__ == 'BernoulliLoss':
            return {
                'ave_tr_losses': self.average_tr_loss_list,
                'ave_val_losses': self.average_val_loss_list,
                'lift_values': self.val_acc_list
            }
        elif self.model.nn_loss.__class__.__name__ == 'FeatureNormalizedMSE':
            return {
                'ave_tr_losses': self.average_tr_loss_list,
                'ave_val_losses': self.average_val_loss_list
            }
        else:
            raise NotImplementedError

    def train(self, n_epoch_training):

        self.logger.log(
            'Number of training epochs: %d' % n_epoch_training,
            'Trainer.train()'
        )
        self.logger.log(
            'Loss function: %s' % str(self.model.nn_loss)[:-3],
            'Trainer.train()'
        )
        self.logger.log(
            'Training starts... \n',
            'Trainer.train()'
        )

        self._init_variables_for_recording()

        self.average_tr_loss_list.append(
            self._train_loss(
                self.count_of_warming_and_training
            )
        )
        if self.model.nn_loss.__class__.__name__ == 'BernoulliLoss':
            val_loss, val_acc = self._epoch_val(self.count_of_warming_and_training)
            self.average_val_loss_list.append(val_loss)
            self.val_acc_list.append(val_acc)
        elif self.model.nn_loss.__class__.__name__ == 'FeatureNormalizedMSE':
            self.average_val_loss_list.append(
                self._epoch_val(self.count_of_warming_and_training)
            )
        else:
            raise NotImplementedError

        for epoch_idx in range(n_epoch_training):

            self.count_of_warming_and_training += 1

            self.average_tr_loss_list.append(
                self._epoch_train(
                    self.count_of_warming_and_training,
                )
            )

            if self.model.nn_loss.__class__.__name__ == 'BernoulliLoss':
                val_loss, val_acc = self._epoch_val(self.count_of_warming_and_training)
                self.average_val_loss_list.append(val_loss)
                self.val_acc_list.append(val_acc)
            elif self.model.nn_loss.__class__.__name__ == 'FeatureNormalizedMSE':
                self.average_val_loss_list.append(
                    self._epoch_val(self.count_of_warming_and_training)
                )
            else:
                raise NotImplementedError

            if self.count_of_warming_and_training % self.train_param['save_model_after_n_epoch'] \
                    == (self.train_param['save_model_after_n_epoch'] - 1):
                index = int(time())
                self.model.save(
                    self.train_param['model_save_path'],
                    index
                )
                self.logger.log(
                    'Saved a model indexed by %d. \n' % index,
                    'Trainer.train()'
                )

        index = int(time())
        self.model.save(
            self.train_param['model_save_path'],
            index
        )
        self.logger.log(
            'Saved a model indexed by %d. \n' % index,
            'Trainer.train()'
        )

        self.logger.log(
            '========Training Finished========',
            'Trainer.train()'
        )

        if self.model.nn_loss.__class__.__name__ == 'BernoulliLoss':
            return {
                'ave_tr_losses': self.average_tr_loss_list,
                'ave_val_losses': self.average_val_loss_list,
                'lift_values': self.val_acc_list
            }
        elif self.model.nn_loss.__class__.__name__ == 'FeatureNormalizedMSE':
            return {
                'ave_tr_losses': self.average_tr_loss_list,
                'ave_val_losses': self.average_val_loss_list
            }
        else:
            raise NotImplementedError

    def detect_anomaly(self, train_labels, val_labels, threshold):

        train_prediction_float_array = self.model.predict(self.train_data)
        train_loss = self.model.nn_loss.individial_loss(
            train_prediction_float_array,
            self.train_target,
            1.0 / self.std_per_feature_float_tensor
        )

        val_prediction_float_array = self.model.predict(self.val_data)
        val_loss = self.model.nn_loss.individial_loss(
            val_prediction_float_array,
            self.val_target,
            1.0 / self.std_per_feature_float_tensor
        )

        train_ind_loss = np.concatenate(
            [np.arange(train_loss.shape[0])[:, np.newaxis], train_loss[:, np.newaxis]],
            axis=1
        )
        val_ind_loss = np.concatenate(
            [np.arange(val_loss.shape[0])[:, np.newaxis], val_loss[:, np.newaxis]],
            axis=1
        )

        train_loss = np.concatenate(
            [train_loss[:, np.newaxis], train_labels],
            axis=1
        )
        val_loss = np.concatenate(
            [val_loss[:, np.newaxis], val_labels],
            axis=1
        )

        sorted_train_loss = train_loss[train_loss[:, 0].argsort()]
        sorted_val_loss = val_loss[val_loss[:, 0].argsort()]

        indxed_sorted_train_loss = np.concatenate(
            [np.arange(sorted_train_loss.shape[0])[:, np.newaxis], sorted_train_loss],
            axis=1
        )
        indxed_sorted_val_loss = np.concatenate(
            [np.arange(sorted_val_loss.shape[0])[:, np.newaxis], sorted_val_loss],
            axis=1
        )

        positive_indexed_sorted_train_loss = indxed_sorted_train_loss[
                                             np.where(indxed_sorted_train_loss[:, 2] == 1.0)[0], :2]
        negative_indexed_sorted_train_loss = indxed_sorted_train_loss[
                                             np.where(indxed_sorted_train_loss[:, 2] == 0.0)[0], :2]
        positive_indexed_sorted_val_loss = indxed_sorted_val_loss[
                                           np.where(indxed_sorted_val_loss[:, 2] == 1.0)[0], :2]
        negative_indexed_sorted_val_loss = indxed_sorted_val_loss[
                                           np.where(indxed_sorted_val_loss[:, 2] == 0.0)[0], :2]

        fig_0 = plt.figure()
        fig_0_ax = fig_0.add_subplot(111)
        fig_0_ax.plot(positive_indexed_sorted_train_loss[:, 0],
                      positive_indexed_sorted_train_loss[:, 1],
                      'rx',
                      label='Positive Samples')
        fig_0_ax.plot(negative_indexed_sorted_train_loss[:, 0],
                      negative_indexed_sorted_train_loss[:, 1],
                      'k.',
                      label='Negative Samples')
        fig_0_ax.plot(np.linspace(0, indxed_sorted_train_loss.shape[0], 100),
                      threshold * np.ones(100),
                      'b--')
        fig_0_ax.legend()
        fig_0_ax.set_title('Train Samples')
        fig_0.savefig('./log/train_samples_anomaly_detection.png')

        fig_1 = plt.figure()
        fig_1_ax = fig_1.add_subplot(111)
        fig_1_ax.plot(positive_indexed_sorted_val_loss[:, 0],
                      positive_indexed_sorted_val_loss[:, 1],
                      'rx',
                      label='Positive Samples')
        fig_1_ax.plot(negative_indexed_sorted_val_loss[:, 0],
                      negative_indexed_sorted_val_loss[:, 1],
                      'k.',
                      label='Negative Samples')
        fig_1_ax.plot(np.linspace(0, indxed_sorted_val_loss.shape[0], 100),
                      threshold * np.ones(100),
                      'b--')
        fig_1_ax.legend()
        fig_1_ax.set_title('Val Samples')
        fig_1.savefig('./log/val_samples_anomaly_detection.png')

        return train_ind_loss[train_ind_loss[:, 1] < threshold, 0].astype('int'), \
               train_ind_loss[train_ind_loss[:, 1] > threshold, 0].astype('int'), \
               val_ind_loss[val_ind_loss[:, 1] < threshold, 0].astype('int'), \
               val_ind_loss[val_ind_loss[:, 1] > threshold, 0].astype('int')
