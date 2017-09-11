from copy import deepcopy
from os import makedirs, listdir, getcwd
from pickle import dump, load
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import xgboost as xgb
from torch.autograd import Variable

from data_loader import DataLoader, ProbWeightedDataLoader
from log_to_file import Logger


class BoostNN:

    hessian_constant = 1e-8

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
        :param xgb_param:           a dictionary of parameters for XGB
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

        if 'path' in kwargs:
            path = kwargs['path']
            if 'index' in kwargs and 'prefix' in kwargs:
                self.load(path, prefix=kwargs['prefix'], index=kwargs['index'])
            elif 'index' in kwargs and not 'prefix' in kwargs:
                self.load(path, index=kwargs['index'])
            elif not 'index' in kwargs and 'prefix' in kwargs:
                self.load(path, prefix=kwargs['prefix'])
            else:
                self.load(path)
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
                    np.random.randint(0, high=100000000)
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

    def _add_booster_layer(self, channel_idx, first_dim_idx, second_dim_idx, lr=None):

        self.xgb_param['seed'] = \
            self.booster_layer_seed_list[channel_idx][first_dim_idx][second_dim_idx]

        if lr is not None:
            self.xgb_param['learning_rate'] = lr

        self.booster_layer_list[channel_idx][first_dim_idx][second_dim_idx].append(
            xgb.Booster(params=self.xgb_param)
        )

        self.booster_layer_count_list[channel_idx][first_dim_idx][second_dim_idx] += 1

        self.newly_updated_booster_layer_list[channel_idx][first_dim_idx][second_dim_idx] = True

    def train_booster_layer(self, dmatrix, grad, hess, lr=None):
        """
        Train a booster layer.

        :type dmatrix:  xgb.core.DMatrix
        :type grad:     torch.FloatTensor
        :type hess:     torch.FloatTensor
        :type lr:       float or double

        :param dmatrix: [
                            batch_size * add_booster_layer_after_n_batch,
                            n_channel_to_boosting * first_dim_to_boosting * second_dim_to_boosting
                        ]
                        on host
        :param grad:    [
                            batch_size * add_booster_layer_after_n_batch,
                            n_channel_from_boosting,
                            first_dim_from_boosting,
                            second_dim_from_boosting
                        ]
                        on host
        :param hess:    [
                            batch_size * add_booster_layer_after_n_batch,
                            n_channel_from_boosting,
                            first_dim_from_boosting,
                            second_dim_from_boosting
                        ]
                        on host
        :param lr:      use learning_rate in xgb_param if not provided
        """

        n_rows = dmatrix.num_row()

        dmatrix.set_base_margin(
            np.zeros(
                n_rows,
                dtype='float32'
            )
        )

        for channel_idx in range(self.n_channel_from_boosting):
            for first_dim_idx in range(self.first_dim_from_boosting):
                for second_dim_idx in range(self.second_dim_from_boosting):

                    abs_ave_of_grad = torch.mean(
                        torch.abs(
                            grad[:, channel_idx, first_dim_idx, second_dim_idx]
                        )
                    )

                    train_grad = torch.div(
                        grad[:, channel_idx, first_dim_idx, second_dim_idx],
                        abs_ave_of_grad
                    )

                    train_hess = torch.div(
                        torch.add(
                            hess[:, channel_idx, first_dim_idx, second_dim_idx],
                            self.hessian_constant
                        ),
                        abs_ave_of_grad
                    )

                    self._add_booster_layer(
                        channel_idx,
                        first_dim_idx,
                        second_dim_idx,
                        lr
                    )

                    self.booster_layer_coef_list[channel_idx][first_dim_idx][second_dim_idx].append(
                        1.0
                    )

                    self.booster_layer_list[channel_idx][first_dim_idx][second_dim_idx][-1].boost(
                        dmatrix,
                        train_grad.numpy().tolist(),
                        train_hess.numpy().tolist()
                    )

        return np.sum(self.newly_updated_booster_layer_list)

    def _grad_hook(self):
        def hook(grad):
            self.grad_of_input_to_nn_model = grad

        return hook

    def nn_forward(self, x, requires_grad=True):
        """
        Forward the net.
        :param torch.FloatTensor x: [
                                        batch_size,
                                        n_channel_from_boosting,
                                        first_dim_from_boosting,
                                        second_dim_from_boosting
                                    ]
        :param bool requires_grad:  requires gradient or not
        :return Variable:           nn output
        """
        self.input_to_nn_model = Variable(x.contiguous(), requires_grad=requires_grad)

        if requires_grad:
            self.input_to_nn_model.register_hook(self._grad_hook())

        outputs = self.nn_model(self.input_to_nn_model)

        return outputs

    def predict(self, x):
        """
        Predict.
        :param np.array x:  [
                                batch_size,
                                n_channel_to_boosting,
                                first_dim_to_boosting,
                                second_dim_to_boosting
                            ]
        :return np.array:   nn output
        """

        n_rows = x.shape[0]

        dmatrix = xgb.DMatrix(
            x.reshape(
                n_rows,
                -1
            )
        )
        dmatrix.set_base_margin(
            np.zeros(
                n_rows,
                dtype='float32'
            )
        )

        xgb_predictions = torch.zeros(
            [
                n_rows,
                self.n_channel_from_boosting,
                self.first_dim_from_boosting,
                self.second_dim_from_boosting
            ],
        )

        for channel_idx in range(self.n_channel_from_boosting):
            for first_dim_idx in range(self.first_dim_from_boosting):
                for second_dim_idx in range(self.second_dim_from_boosting):
                    xgb_predictions[:, channel_idx, first_dim_idx, second_dim_idx] = \
                        self._booster_predict(
                            dmatrix,
                            channel_idx,
                            first_dim_idx,
                            second_dim_idx
                        )

        predictions = self.nn_forward(xgb_predictions, requires_grad=False)

        return predictions.data.numpy()

    def _booster_predict(self, dmatrix, channel_idx, first_dim_idx, second_dim_idx):

        predictions = torch.zeros(dmatrix.num_row())

        for booster_layer_idx in range(self.booster_layer_count_list[channel_idx][first_dim_idx][second_dim_idx]):
            predictions += (
                torch.from_numpy(
                    self.booster_layer_list
                    [channel_idx]
                    [first_dim_idx]
                    [second_dim_idx]
                    [booster_layer_idx].predict(dmatrix)
                )
                *
                self.booster_layer_coef_list[channel_idx][first_dim_idx][second_dim_idx][booster_layer_idx]
            )

        return predictions

    def predict_with_newly_added_booster(self, dmatrix, reset_newly_updated_booster_layer_list):
        """
        :type dmatrix:                                  xgb.core.DMatrix
        :type reset_newly_updated_booster_layer_list:   bool
        :rtype:                                         torch.FloatTensor
        """

        predictions = torch.zeros(
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

                        predictions[:, channel_idx, first_dim_idx, second_dim_idx] = (
                            torch.from_numpy(
                                self.booster_layer_list[channel_idx][first_dim_idx][second_dim_idx][-1]
                                    .predict(dmatrix)
                            )
                            *
                            self.booster_layer_coef_list[channel_idx][first_dim_idx][second_dim_idx][-1]
                        )

                        if reset_newly_updated_booster_layer_list:
                            self.newly_updated_booster_layer_list[channel_idx][first_dim_idx][second_dim_idx] = False

        return predictions

    def save(self, path, prefix=None, index=None):
        if index and prefix:
            file_name = '%s/%s/%s_bnn_%d.dat' % (getcwd(), path, prefix, index)
        elif index and not prefix:
            file_name = '%s/%s/bnn_%d.dat' % (getcwd(), path, index)
        elif not index and prefix:
            file_name = '%s/%s/%s_bnn.dat' % (getcwd(), path, prefix)
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

    def load(self, path, prefix=None, index=None):
        if index and prefix:
            file_name = '%s/%s_bnn_%d.dat' % (path, prefix, index)
        elif index and not prefix:
            file_name = '%s/bnn_%d.dat' % (path, index)
        elif not index and prefix:
            file_name = '%s/%s_bnn.dat' % (path, prefix)
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


class Trainer:
    def __init__(self, model, train_param):

        # initialize a logger
        self.logger = Logger(
            'main',
            to_stdout=train_param['verbose'],
            path=train_param['log_path']
        )
        self.logger.log(
            'Initialization Starts...',
            'Trainer.__init__()'
        )
        self.logger.log(
            'NN Name: ' + model.nn_model.__class__.__name__,
            'Trainer.__init__()'
        )
        self.logger.log(
            'NN Structure: \n' + model.nn_model.__repr__(),
            'Trainer.__init__()'
        )

        self.logger.log(
            '================= XGBoost Parameters ==============',
            'Trainer.__init__()'
        )
        for key in model.xgb_param:
            self.logger.log(
                key + ': ' + str(model.xgb_param[key] if key is not None else ' '),
                'Trainer.__init__()'
            )
        self.logger.log(
            '===================================================',
            'Trainer.__init__()'
        )

        self.logger.log(
            '============ Neural Networks Parameters ===========',
            'Trainer.__init__()'
        )
        for key in model.nn_param:
            self.logger.log(
                key + ': ' + str(model.nn_param[key]),
                'Trainer.__init__()'
            )
        self.logger.log(
            '===================================================',
            'Trainer.__init__()'
        )

        self.logger.log(
            '=============== Training Parameters ===============',
            'Trainer.__init__()'
        )
        for key in train_param:
            self.logger.log(
                key + ': ' + str(train_param[key]),
                'Trainer.__init__()'
            )
        self.logger.log(
            '===================================================',
            'Trainer.__init__()'
        )

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

        # for data
        self.train_data = None
        self.train_target = None
        self.val_data = None
        self.val_target = None

        self.train_e_exp = None
        self.train_weight = None
        self.val_e_exp = None
        self.val_weight = None

        self.train_set_loader = None
        self.val_set_loader = None

        # for adam
        self.ma_grad_float_tensor = None
        self.ma_hess_float_tensor = None
        self.count_of_booster_layer_training = 0

        # for recording
        self.average_tr_loss_list = None
        self.average_val_loss_list = None

        self.grad_scale_list = None
        self.hess_scale_list = None

        self.val_acc_list = None

        self.count_of_warming_and_training = -1

        self.logger.log(
            'Initialization Ends...\n',
            'Trainer.__init__()'
        )

    @staticmethod
    def _start_string(phase, output_0, output_1, output_2, output_3, output_4):
        return ' ' + phase + ' %6d, Batch [%3d/%3d], max No. of boosters added: %3d, time: %.4f' \
                             % (output_0, output_1, output_2, output_3, output_4)

    @staticmethod
    def _blank_spaces(phase):
        return ' ' * (' ' + phase + ' %6d, ' % 0).__len__()

    def input_data(self, train_data, train_target, val_data, val_target, make_copy_of_data=False,
                   train_e_exp=None, train_weight=None, val_e_exp=None, val_weight=None):

        if make_copy_of_data:
            self.train_data = deepcopy(train_data)
            self.train_target = deepcopy(train_target)
            self.val_data = deepcopy(val_data)
            self.val_target = deepcopy(val_target)
            self.train_e_exp = deepcopy(train_e_exp)
            self.train_weight = deepcopy(train_weight)
            self.val_e_exp = deepcopy(val_e_exp)
            self.val_weight = deepcopy(val_weight)
        else:
            self.train_data = train_data
            self.train_target = train_target
            self.val_data = val_data
            self.val_target = val_target
            self.train_e_exp = train_e_exp
            self.train_weight = train_weight
            self.val_e_exp = val_e_exp
            self.val_weight = val_weight

        self.logger.log(
            'train_data max: ' + str(self.train_data[~np.isnan(self.train_data)].max()) +
            '\t\ttrain_data min:  ' + str(self.train_data[~np.isnan(self.train_data)].min()) +
            '\t\tval_data max:  ' + str(self.val_data[~np.isnan(self.val_data)].max()) +
            '\t\tval_data min:  ' + str(self.val_data[~np.isnan(self.val_data)].min()),
            'Trainer.input_data()'
        )

        self.std_per_feature_float_tensor = self._per_feature_operation(
            self.train_data,
            np.std,
            self.logger
        )

        # put data into a DataLoader
        self.train_set_loader = ProbWeightedDataLoader(
            self.train_data,
            self.train_target,
            self.train_param['batch_size'],
            [
                self.model.n_channel_from_boosting,
                self.model.first_dim_from_boosting,
                self.model.second_dim_from_boosting
            ],
            e_exp=self.train_e_exp,
            weight=self.train_weight,
            train_param=self.train_param
        )

        self.val_set_loader = DataLoader(
            self.val_data,
            self.val_target,
            self.val_data.shape[0],
            [
                self.model.n_channel_from_boosting,
                self.model.first_dim_from_boosting,
                self.model.second_dim_from_boosting
            ],
            e_exp=self.val_e_exp,
            weight=self.val_weight,
            train_param=self.train_param
        )
        self.logger.log(
            'train_set_loader and val_set_loader DONE.\n',
            'Trainer.input_data()'
        )

    @staticmethod
    def _per_feature_operation(data, fun, logger=None):
        sample_n = data.shape[0]
        result = np.array(
            [
                [
                    fun(
                        item[
                            ~np.isnan(item)
                        ]
                    )
                ]
                if np.sum(np.isnan(item)) < sample_n else 1.0
                for item in data.reshape(data.shape[0], -1).transpose()
            ]
        )
        result = result.astype('float32')
        if logger:
            range_string = 'feature ' + fun.__name__ + ': \n'
            for idx, item in enumerate(result):
                range_string += ' ' * 10 + 'f_' + str(idx) + ': %.4f\t' % item
                if idx % 5 == 4:
                    range_string += '\n'
            # logger.log(range_string, 'Trainer._per_feature_operation()')
        return torch.from_numpy(result)

    def _update_ma(self, grad_float_tensor, hess_float_tensor):
        """
        Update ADAM.
        :param torch.FloatTensor grad_float_tensor: gradients
        :param torch.FloatTensor hess_float_tensor: Hessian
        """

        if self.ma_grad_float_tensor is None or self.ma_hess_float_tensor is None:
            self.ma_grad_float_tensor = (1.0 - self.grad_ma_coef) * grad_float_tensor
            self.ma_hess_float_tensor = (1.0 - self.hess_ma_coef) * hess_float_tensor
        else:
            self.ma_grad_float_tensor = (
                self.grad_ma_coef * self.ma_grad_float_tensor
                +
                (1.0 - self.grad_ma_coef) * grad_float_tensor
            )
            self.ma_hess_float_tensor = (
                self.hess_ma_coef * self.ma_hess_float_tensor
                + (1.0 - self.hess_ma_coef) * hess_float_tensor
            )

    def _nn_step(self, margin_float_tensor, target_float_tensor, e_exp_float_tensor, sample_weight_float_tensor, mode):
        """
        :type margin_float_tensor: torch.FloatTensor
        :type target_float_tensor: torch.FloatTensor
        :type e_exp_float_tensor: torch.FloatTensor
        :type sample_weight_float_tensor: torch.FloatTensor
        """

        target_float_variable = Variable(target_float_tensor)
        e_exp_float_variable = Variable(e_exp_float_tensor)
        sample_weight_float_variable = Variable(sample_weight_float_tensor)
        feature_weight_float_variable = Variable(1.0 / self.std_per_feature_float_tensor)

        if mode == 'loss_and_backward_and_update':

            self.model.nn_optimizer.zero_grad()

            output_float_variable = self.model.nn_forward(
                margin_float_tensor,
                requires_grad=True
            )

            corrected_target_float_variable = self._correct_target(
                output_float_variable,
                target_float_variable,
                e_exp_float_variable
            )

            loss_variable, display_loss_variable = self._cal_loss(
                output_float_variable,
                target_float_variable,
                e_exp_float_variable,
                sample_weight_float_variable,
                feature_weight_float_variable,
                corrected_target_float_variable,
            )

            loss_variable.backward()

            self.model.nn_optimizer.step()

            return loss_variable.data[0], display_loss_variable.data[0]

        elif mode == 'loss_and_backward':

            self.model.nn_optimizer.zero_grad()

            output_float_variable = self.model.nn_forward(
                margin_float_tensor,
                requires_grad=True
            )

            corrected_target_float_variable = self._correct_target(
                output_float_variable,
                target_float_variable,
                e_exp_float_variable
            )

            loss_variable, display_loss_variable = self._cal_loss(
                output_float_variable,
                target_float_variable,
                e_exp_float_variable,
                sample_weight_float_variable,
                feature_weight_float_variable,
                corrected_target_float_variable
            )

            loss_variable.backward()

            return loss_variable.data[0], \
                display_loss_variable.data[0], \
                output_float_variable.data, \
                corrected_target_float_variable.data

        elif mode == 'loss':

            output_float_variable = self.model.nn_forward(
                margin_float_tensor,
                requires_grad=False
            )

            display_loss_variable = self._cal_loss(
                output_float_variable,
                target_float_variable,
                e_exp_float_variable,
                sample_weight_float_variable,
                feature_weight_float_variable
            )

            return display_loss_variable.data[0], output_float_variable.data

    def _xgb_step(self, data_float_tensor, grad_float_tensor, hess_float_tensor, lr=None):
        """
        Add a booster layer for those non-zero gradients.
        :param torch.FloatTensor data_float_tensor: [
                                                        batch_size * add_booster_layer_after_n_batch,
                                                        n_channel_to_boosting
                                                        * first_dim_to_boosting
                                                        * second_dim_to_boosting
                                                    ]
        :param torch.FloatTensor grad_float_tensor: [
                                                        batch_size * add_booster_layer_after_n_batch,
                                                        n_channel_from_boosting,
                                                        first_dim_from_boosting,
                                                        second_dim_from_boosting
                                                    ]
        :param torch.FloatTensor hess_float_tensor: [
                                                        batch_size * add_booster_layer_after_n_batch,
                                                        n_channel_from_boosting,
                                                        first_dim_from_boosting,
                                                        second_dim_from_boosting
                                                    ]
        :param float or double lr:                  use learning_rate in xgb_param if not provided
        """

        self._update_ma(grad_float_tensor, hess_float_tensor)

        train_double_dmatrix = xgb.DMatrix(
            data_float_tensor.view(
                data_float_tensor.size()[0],
                -1
            ).cpu().numpy()
        )

        train_grad_float_tensor = (
            self.ma_grad_float_tensor.cpu()
            /
            (1.0 - self.grad_ma_coef ** (self.count_of_booster_layer_training + 1.0))
        )

        train_hess_float_tensor = (
            self.ma_hess_float_tensor.cpu()
            /
            (1.0 - self.hess_ma_coef ** (self.count_of_booster_layer_training + 1.0))
        )

        n_new_trees = self.model.train_booster_layer(
            train_double_dmatrix,
            train_grad_float_tensor,
            train_hess_float_tensor,
            lr
        )

        self.count_of_booster_layer_training += 1

        self.logger.log(
            'Added new trees for %d nodes.' % n_new_trees,
            'Trainer._xgb_step()'
        )

    def _correct_target(self, output_float_variable, target_float_variable, e_exp_float_variable):
        """
        Correct outputs
        :param Variable output_float_variable:  [
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
        :param Varaible target_float_variable:  [
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
        :return Variable:                       of the same shape of output and target
        """

        if self.target_regularization_coef == 1.0:
            return target_float_variable

        else:
            corrected_target_float_variable = (
                self.target_regularization_coef
                * target_float_variable
                +
                (1.0 - self.target_regularization_coef)
                * self.model.nn_loss.torch_link(output_float_variable, e_exp_float_variable)
            )
            return corrected_target_float_variable

    def _cal_loss(self,
                  output_float_variable,
                  target_float_variable,
                  e_exp_float_variable,
                  sample_weight_float_variable,
                  feature_weight_float_variable,
                  corrected_target_float_variable=None
                  ):

        loss_variable = None

        if corrected_target_float_variable is not None:
            loss_variable = self.model.nn_loss(
                output_float_variable,
                corrected_target_float_variable,
                e_exp_float_variable,
                sample_weight_float_variable,
                feature_weight_float_variable
            )

        display_loss_variable = self.model.nn_loss(
            output_float_variable,
            target_float_variable,
            e_exp_float_variable,
            sample_weight_float_variable,
            feature_weight_float_variable
        )

        if corrected_target_float_variable:
            return loss_variable, display_loss_variable
        else:
            return display_loss_variable

    def _cal_grad_and_hess(self,
                           output_float_tensor,
                           corrected_target_float_tensor,
                           e_exp_float_tensor,
                           weight_float_tensor
                           ):

        grad_float_tensor = \
            self.model \
                .grad_of_input_to_nn_model \
                .data

        hess_flaot_tensor = self.model.nn_loss.hess(
            grad_float_tensor,
            output_float_tensor,
            corrected_target_float_tensor,
            e_exp_float_tensor,
            weight_float_tensor
        )

        return grad_float_tensor, hess_flaot_tensor

    def _epoch_train(self, epoch):

        epoch_start_time = time()

        running_loss = 0.0
        running_display_loss = 0.0

        self.train_set_loader.start_new_round()

        while self.train_set_loader.next_batch():

            time_start = time()

            margin_float_tensor, target_float_tensor, e_exp_float_tensor, weight_float_tensor = \
                self.train_set_loader.get_data_for_nn(self.model.enable_cuda)
            loss, display_loss = self._nn_step(
                margin_float_tensor,
                target_float_tensor,
                e_exp_float_tensor,
                weight_float_tensor,
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
                + 'corrected batch train loss: %.24lf' %
                (
                    loss / self.train_set_loader.batch_size,
                ),
                'Trainer._epoch_train()'
            )

            self.logger.log(
                self._blank_spaces(
                    'Epoch'
                )
                + 'original batch train loss: %.24lf' %
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

                data_float_tensor, _, target_float_tensor, e_exp_float_tensor, weight_float_tensor = \
                    self.train_set_loader.get_data_for_booster_layer(self.model.enable_cuda)

                for booster_layer_idx in range(self.train_param['n_booster_layer_to_add']):
                    time_start = time()

                    _, margin_float_tensor, _, _, _ = \
                        self.train_set_loader.get_data_for_booster_layer(self.model.enable_cuda)

                    loss, display_loss, output_float_tensor, corrected_target_float_tensor = self._nn_step(
                        margin_float_tensor,
                        target_float_tensor,
                        e_exp_float_tensor,
                        weight_float_tensor,
                        'loss_and_backward'
                    )

                    grad_float_tensor, hess_float_tensor = self._cal_grad_and_hess(
                        output_float_tensor,
                        corrected_target_float_tensor,
                        e_exp_float_tensor,
                        weight_float_tensor,
                    )

                    self.grad_scale_list.append(torch.mean(torch.abs(grad_float_tensor)))
                    self.hess_scale_list.append(torch.mean(torch.abs(hess_float_tensor)))

                    self._xgb_step(
                        data_float_tensor,
                        grad_float_tensor,
                        hess_float_tensor
                    )

                    train_margin_update = self.model.predict_with_newly_added_booster(
                        xgb.DMatrix(
                            self.train_data.reshape(
                                self.train_data.shape[0],
                                -1
                            )
                        ),
                        False
                    )

                    self.train_set_loader.update_margin(
                        train_margin_update,
                        epoch + 1,
                        self.model.nn_loss.torch_link,
                        self.model
                    )

                    self.val_set_loader.update_margin(
                        self.model.predict_with_newly_added_booster(
                            xgb.DMatrix(
                                self.val_data.reshape(
                                    self.val_data.shape[0],
                                    -1
                                )
                            ),
                            True
                        ),
                        epoch + 1,
                        self.model.nn_loss.torch_link,
                        self.model
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
                        + 'L1 norm of gradients: %.24lf' %
                        (
                            self.grad_scale_list[-1],
                        ),
                        'Trainer._epoch_train()'
                    )

                    self.logger.log(
                        self._blank_spaces(
                            'Epoch'
                        )
                        + 'L1 norm of Hessian: %.24lf' %
                        (
                            self.hess_scale_list[-1]
                        ),
                        'Trainer._epoch_train()'
                    )

                    self.logger.log(
                        self._blank_spaces(
                            'Epoch',
                        )
                        + 'L1 norm of %d values of margin update: %.24lf' %
                        (
                            np.prod(self.train_set_loader.margin.shape),
                            float(torch.mean(torch.abs(train_margin_update)))
                        ),
                        'Trainer._epoch_train()'
                    )

                    self.logger.log(
                        self._blank_spaces(
                            'Epoch',
                        )
                        + 'corrected batch train loss: %.24lf' %
                        (
                            loss / self.train_set_loader.get_length_of_data_for_booster_layer(),
                        ),
                        'Trainer._epoch_train()'
                    )

                    self.logger.log(
                        self._blank_spaces(
                            'Epoch',
                        )
                        + 'original batch train loss: %.24lf' %
                        (
                            display_loss / self.train_set_loader.get_length_of_data_for_booster_layer()
                        ),
                        'Trainer._epoch_train()'
                    )

                self.train_set_loader.reset_used_data()

        average_running_loss = running_loss / self.train_set_loader.total_sample_n
        average_running_display_loss = running_display_loss / self.train_set_loader.total_sample_n

        self.logger.log(
            ' Epoch %6d ==>> average train loss: %.24lf' %
            (
                epoch + 1,
                average_running_display_loss
            ),
            'Trainer._epoch_train()'
        )
        self.logger.log(
            '                   average corrected train loss: %.24lf' %
            average_running_loss,
            'Trainer._epoch_train()'
        )
        self.logger.log(
            '                   elapse time: %16.8lf' %
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
        e_exp_float_array_list = None
        if self.model.nn_loss.__class__.__name__ == 'BernoulliLoss':
            target_float_array_list = []
            output_float_array_list = []
            e_exp_float_array_list = []

        self.val_set_loader.start_new_round(shuffle=False)

        while self.val_set_loader.next_batch():

            margin_float_tensor, target_float_tensor, e_exp_float_tensor, weight_float_tensor = \
                self.val_set_loader.get_data_for_nn(self.model.enable_cuda)
            display_loss, output_float_tensor = self._nn_step(
                margin_float_tensor,
                target_float_tensor,
                e_exp_float_tensor,
                weight_float_tensor,
                'loss'
            )

            if self.model.nn_loss.__class__.__name__ == 'BernoulliLoss':
                target_float_array_list.append(target_float_tensor)
                output_float_array_list.append(output_float_tensor)
                e_exp_float_array_list.append(e_exp_float_tensor)

            running_display_loss += display_loss

        average_running_display_loss = running_display_loss / self.val_set_loader.total_sample_n

        self.logger.log(
            ' Epoch %6d ==>> average val loss: %.24lf' %
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

            total_target = torch.cat(target_float_array_list, 0)
            total_output = torch.cat(output_float_array_list, 0)
            total_e_exp = torch.cat(e_exp_float_array_list, 0)

            pct = self.model.nn_loss.lift_level(
                total_output,
                total_target,
                self.train_param['lift_level_at'],
                e_exp=total_e_exp
            )
            self.logger.log(
                ' Epoch %6d ==>> Lift achieved for %.5f %% of top predictions: %.4f %%' %
                (
                    epoch + 1,
                    self.train_param['lift_level_at'] * 100.0,
                    pct * 100.0
                ),
                'Trainer._epoch_val()'
            )

            x_float_tensor, lift_float_tensor = self.model.nn_loss.lift_plot(
                total_output,
                total_target,
                e_exp=total_e_exp
            )

            x_string = ''
            for x in x_float_tensor:
                x_string += '%.5f\t' % x
            lift_string = ''
            for lift in lift_float_tensor:
                lift_string += '%.5f\t' % lift

            self.logger.log(
                ' Epoch %6d ==>> Lift Plot x: %s' %
                (
                    epoch + 1,
                    x_string
                ),
                'Trainer._epoch_val()'
            )

            self.logger.log(
                ' Epoch %6d ==>> Lift Plot y: %s \n' %
                (
                    epoch + 1,
                    lift_string
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
            margin_float_tensor, target_float_tensor, e_exp_float_tensor, weight_float_tensor = \
                self.train_set_loader.get_data_for_nn(self.model.enable_cuda)

            display_loss, _ = self._nn_step(
                margin_float_tensor,
                target_float_tensor,
                e_exp_float_tensor,
                weight_float_tensor,
                'loss'
            )

            running_display_loss += display_loss

        average_running_display_loss = running_display_loss / self.train_set_loader.total_sample_n

        self.logger.log(
            ' Epoch %6d ==>> average train loss: %.24lf' %
            (
                epoch + 1,
                average_running_display_loss
            ),
            'Trainer._train_loss()'
        )
        self.logger.log(
            '                   elapse time: %16.8lf' %
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

            add_booster_layer = \
                self.train_set_loader.current_batch_idx \
                % self.train_param['add_booster_layer_after_n_batch'] \
                == self.train_param['add_booster_layer_after_n_batch'] - 1

            if add_booster_layer:
                time_start = time()

                data_float_tensor, margin_float_tensor, target_float_tensor, e_exp_float_tensor, weight_float_tensor = \
                    self.train_set_loader.get_data_for_booster_layer(self.model.enable_cuda)

                loss, display_loss, output_float_tensor, corrected_target_float_tensor = self._nn_step(
                    margin_float_tensor,
                    target_float_tensor,
                    e_exp_float_tensor,
                    weight_float_tensor,
                    'loss_and_backward'
                )

                running_loss += loss
                running_display_loss += display_loss

                grad_float_tensor, hess_float_tensor = self._cal_grad_and_hess(
                    output_float_tensor,
                    corrected_target_float_tensor,
                    e_exp_float_tensor,
                    weight_float_tensor
                )

                self.grad_scale_list.append(torch.mean(torch.abs(grad_float_tensor)))
                self.hess_scale_list.append(torch.mean(torch.abs(hess_float_tensor)))

                if 'warming_up_lr' in self.model.xgb_param:
                    self._xgb_step(
                        data_float_tensor,
                        grad_float_tensor,
                        hess_float_tensor,
                        self.model.xgb_param['warming_up_lr']
                    )
                else:
                    self._xgb_step(
                        data_float_tensor,
                        grad_float_tensor,
                        hess_float_tensor
                    )

                train_margin_update = self.model.predict_with_newly_added_booster(
                    xgb.DMatrix(
                        self.train_data.reshape(
                            self.train_data.shape[0],
                            -1
                        )
                    ),
                    False
                )

                self.train_set_loader.update_margin(
                    train_margin_update,
                    epoch + 1,
                    self.model.nn_loss.torch_link,
                    self.model
                )

                self.val_set_loader.update_margin(
                    self.model.predict_with_newly_added_booster(
                        xgb.DMatrix(
                            self.val_data.reshape(
                                self.val_data.shape[0],
                                -1
                            )
                        ),
                        True
                    ),
                    epoch + 1,
                    self.model.nn_loss.torch_link,
                    self.model
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
                    + 'L1 norm of gradients: %.24lf' %
                    (
                        self.grad_scale_list[-1],
                    ),
                    'Trainer._epoch_warming_up()'
                )

                self.logger.log(
                    self._blank_spaces(
                        'Warming-Up'
                    )
                    + 'L1 norm of Hessian: %.24lf' %
                    (
                        self.hess_scale_list[-1]
                    ),
                    'Trainer._epoch_warming_up()'
                )

                self.logger.log(
                    self._blank_spaces(
                        'Warming-Up',
                    )
                    + 'L1 norm of %d values of margin update: %.24lf' %
                    (
                        np.prod(self.train_set_loader.margin.shape),
                        float(torch.mean(torch.abs(train_margin_update)))
                    ),
                    'Trainer._epoch_warming_up()'
                )

                self.logger.log(
                    self._blank_spaces(
                        'Warming-Up',
                    )
                    + 'corrected batch train loss: %.24lf' %
                    (
                        loss / self.train_set_loader.get_length_of_data_for_booster_layer(),
                    ),
                    'Trainer._epoch_warming_up()'
                )

                self.logger.log(
                    self._blank_spaces(
                        'Warming-Up',
                    )
                    + 'original batch train loss: %.24lf' %
                    (
                        loss / self.train_set_loader.get_length_of_data_for_booster_layer(),
                    ),
                    'Trainer._epoch_warming_up()'
                )

                self.train_set_loader.reset_used_data()

        average_running_loss = running_loss / self.train_set_loader.total_sample_n
        average_running_display_loss = running_display_loss / self.train_set_loader.total_sample_n

        self.logger.log(
            ' Warming-Up %6d ==>> average train loss: %.24lf' %
            (
                epoch + 1,
                average_running_display_loss
            ),
            'Trainer._epoch_warming_up()'
        )
        self.logger.log(
            '                        average corrected train loss: %.24lf' %
            average_running_loss,
            'Trainer._epoch_warming_up()'
        )
        self.logger.log(
            '                        elapse time: %16.8lf' %
            (
                time() - epoch_start_time
            ),
            'Trainer._epoch_warming_up()'
        )

        return average_running_display_loss

    def _init_variables_for_recording(self):

        if self.count_of_booster_layer_training is None:
            self.count_of_booster_layer_training = 0

        if self.average_tr_loss_list is None:
            self.average_tr_loss_list = []
        if self.average_val_loss_list is None:
            self.average_val_loss_list = []

        if self.grad_scale_list is None:
            self.grad_scale_list = []
        if self.hess_scale_list is None:
            self.hess_scale_list = []

        if self.val_acc_list is None:
            if self.model.nn_loss.__class__.__name__ == 'BernoulliLoss':
                self.val_acc_list = []
            elif self.model.nn_loss.__class__.__name__ == 'FeatureNormalizedMSE':
                pass
            else:
                raise NotImplementedError

    def warming_up(self, n_epoch_warming_up):

        self.logger.log(
            '========= Warming-Up Starts =========',
            'Trainer.warming_up()'
        )

        self.logger.log(
            'Number of warming-up epochs: %d' % n_epoch_warming_up,
            'Trainer.warming_up()'
        )
        self.logger.log(
            'Loss function: %s' % str(self.model.nn_loss)[:-3],
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
            '======== Warming-Up Finished ========\n',
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
            '========== Training Starts ==========',
            'Trainer.train()'
        )
        self.logger.log(
            'Number of training epochs: %d' % n_epoch_training,
            'Trainer.train()'
        )
        self.logger.log(
            'Loss function: %s' % str(self.model.nn_loss)[:-3],
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

            if (
                    self.count_of_warming_and_training
                    %
                    self.train_param['save_model_after_n_epoch']
                    ==
                    (self.train_param['save_model_after_n_epoch'] - 1)
            ):
                index = int(time())
                self.model.save(
                    self.train_param['model_save_path'],
                    self.logger.get_training_start_time(),
                    index
                )
                self.logger.log(
                    'Saved a model indexed by %d.' % index,
                    'Trainer.train()'
                )

        index = int(time())
        self.model.save(
            self.train_param['model_save_path'],
            self.logger.get_training_start_time(),
            index
        )
        self.logger.log(
            'Saved a model indexed by %d.' % index,
            'Trainer.train()'
        )

        self.logger.log(
            '========= Training Finished =========\n\n',
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

    def margin_histogram(self, image_prefix='mh', n_bins=20):

        margin = self.train_set_loader.margin.transpose(1, 2, 3, 0)

        for channel_idx, channel_margin in enumerate(margin):

            fig = plt.figure(figsize=(16, 14))

            for first_dim_idx, first_dim_margin in enumerate(channel_margin):
                for second_dim_idx, second_dim_margin in enumerate(first_dim_margin):
                    y, x = np.histogram(second_dim_margin, bins=n_bins)

                    image_idx = (
                        first_dim_idx * channel_margin.__len__()
                        + second_dim_idx + 1
                    )
                    subplot_idx = (
                        channel_margin.__len__() * 10 ** (len(str(image_idx)) + 1)
                        +
                        first_dim_margin.__len__() * 10 ** len(str(image_idx))
                        +
                        image_idx
                    )

                    # print(channel_margin.__len__() * 10 ** (len(str(image_idx)) + 1),
                    #       first_dim_margin.__len__() * 10 ** len(str(image_idx)),
                    #       subplot_idx)

                    ax = fig.add_subplot(subplot_idx)
                    ax.bar(x[:-1], y, x[1] - x[0])
                    ax.set_title(
                        '(%d, %d, %d)' %
                        (
                            channel_idx,
                            first_dim_idx,
                            second_dim_idx
                        )
                    )

            fig.savefig(
                '%s/%s/%s_%d.png' % (
                    getcwd(),
                    self.train_param['log_path'],
                    image_prefix,
                    channel_idx
                )
            )

    def detect_anomaly(self, train_labels, val_labels, threshold):

        train_prediction_float_array = self.model.predict(self.train_data)
        train_loss = self.model.nn_loss.individual_loss(
            torch.from_numpy(train_prediction_float_array),
            torch.from_numpy(self.train_set_loader.target),
            torch.from_numpy(np.ones([self.train_set_loader.target.shape[0], 1], dtype='float32')),
            torch.from_numpy(np.ones([self.train_set_loader.target.shape[0], 1], dtype='float32')),
            1.0 / self.std_per_feature_float_tensor
        ).cpu().numpy()

        val_prediction_float_array = self.model.predict(self.val_data)
        val_loss = self.model.nn_loss.individual_loss(
            torch.from_numpy(val_prediction_float_array),
            torch.from_numpy(self.val_set_loader.target),
            torch.from_numpy(np.ones([self.val_set_loader.target.shape[0], 1], dtype='float32')),
            torch.from_numpy(np.ones([self.val_set_loader.target.shape[0], 1], dtype='float32')),
            1.0 / self.std_per_feature_float_tensor
        ).cpu().numpy()

        train_ind_loss = np.concatenate(
            [np.arange(train_loss.shape[0])[:, np.newaxis], train_loss],
            axis=1
        )
        val_ind_loss = np.concatenate(
            [np.arange(val_loss.shape[0])[:, np.newaxis], val_loss],
            axis=1
        )

        train_loss = np.concatenate(
            [train_loss, train_labels],
            axis=1
        )
        val_loss = np.concatenate(
            [val_loss, val_labels],
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

        positive_indexed_sorted_train_loss = \
            indxed_sorted_train_loss[np.where(indxed_sorted_train_loss[:, 2] == 1.0)[0], :2]
        negative_indexed_sorted_train_loss = \
            indxed_sorted_train_loss[np.where(indxed_sorted_train_loss[:, 2] == 0.0)[0], :2]
        positive_indexed_sorted_val_loss = \
            indxed_sorted_val_loss[np.where(indxed_sorted_val_loss[:, 2] == 1.0)[0], :2]
        negative_indexed_sorted_val_loss = \
            indxed_sorted_val_loss[np.where(indxed_sorted_val_loss[:, 2] == 0.0)[0], :2]

        if 'log' not in listdir(getcwd()):
            makedirs('log')

        fig_0 = plt.figure()
        fig_0_ax = fig_0.add_subplot(111)
        fig_0_ax.plot(
            positive_indexed_sorted_train_loss[:, 0],
            positive_indexed_sorted_train_loss[:, 1],
            'rx',
            label='Positive Samples'
        )
        fig_0_ax.plot(
            negative_indexed_sorted_train_loss[:, 0],
            negative_indexed_sorted_train_loss[:, 1],
            'k.',
            label='Negative Samples'
        )
        fig_0_ax.plot(
            np.linspace(
                0,
                indxed_sorted_train_loss.shape[0],
                100
            ),
            threshold * np.ones(100),
            'b--'
        )
        fig_0_ax.legend()
        fig_0_ax.set_title('Train Samples')
        fig_0.savefig('./log/train_samples_anomaly_detection.png')

        fig_1 = plt.figure()
        fig_1_ax = fig_1.add_subplot(111)
        fig_1_ax.plot(
            positive_indexed_sorted_val_loss[:, 0],
            positive_indexed_sorted_val_loss[:, 1],
            'rx',
            label='Positive Samples'
        )
        fig_1_ax.plot(
            negative_indexed_sorted_val_loss[:, 0],
            negative_indexed_sorted_val_loss[:, 1],
            'k.',
            label='Negative Samples'
        )
        fig_1_ax.plot(
            np.linspace(
                0,
                indxed_sorted_val_loss.shape[0],
                100
            ),
            threshold * np.ones(100),
            'b--'
        )
        fig_1_ax.legend()
        fig_1_ax.set_title('Val Samples')
        fig_1.savefig('./log/val_samples_anomaly_detection.png')

        return train_ind_loss[train_ind_loss[:, 1] < threshold, 0].astype('int'), \
               train_ind_loss[train_ind_loss[:, 1] > threshold, 0].astype('int'), \
               val_ind_loss[val_ind_loss[:, 1] < threshold, 0].astype('int'), \
               val_ind_loss[val_ind_loss[:, 1] > threshold, 0].astype('int')