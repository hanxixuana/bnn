#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class FeatureNormalizedMSE(nn.Module):
    def __init__(self):
        super(FeatureNormalizedMSE, self).__init__()

    @staticmethod
    def link(output_float_variable):
        return output_float_variable

    def forward(self, outputs, targets, feature_weights):
        """
        Nan-omitted feature-weighted MSE
        :param outputs: a two-dimensional Variable
        :param targets: a two-dimensional Variable
        :param feature_weights: a one-dimensional Tensor
        :return: a loss
        """

        targets_array = targets.data.numpy()
        num_position = ~np.isnan(targets_array)
        num_position_variable = Variable(torch.from_numpy(num_position.astype('float')).float())

        # replace nan in targets with 0
        targets_array[~num_position] = 0.0

        residual = (targets - outputs) * num_position_variable
        residual = residual.view(residual.size()[0], -1)
        standardized_residual = torch.mm(residual, Variable(torch.diag(feature_weights)))
        average_standardized_residual = \
            torch.sum(
                standardized_residual * standardized_residual,
                1
            ) \
            / \
            torch.sum(
                num_position_variable.view(
                    num_position_variable.size()[0],
                    -1
                ),
                1
            )

        return torch.sum(average_standardized_residual)

    @staticmethod
    def hess(grad_long_double_array):
        """
        Calculate Hessian diagonal line.
        :param grad_long_double_array:          a numpy array of the shape of
                                                [
                                                    batch_size * add_booster_layer_after_n_batch,
                                                    n_channel_from_boosting,
                                                    first_dim_from_boosting,
                                                    second_dim_from_boosting
                                                ]
        :return:                                a numpy array of the shape of
                                                [
                                                    batch_size * add_booster_layer_after_n_batch,
                                                    n_channel_from_boosting,
                                                    first_dim_from_boosting,
                                                    second_dim_from_boosting
                                                ]
        """
        return np.ones(grad_long_double_array.shape)

    @staticmethod
    def individial_loss(outputs, targets, feature_weights):
        """
        Nan-omitted feature-weighted individual MSE
        :param outputs: a 4-dimensional numpy array
        :param targets: a 4-dimensional numpy array
        :param feature_weights: a one-dimensional Tensor
        :return: individual losses
        """
        targets_array = targets.copy()
        num_position = ~np.isnan(targets_array)
        num_position_float_array = num_position.astype('float')

        # replace nan in targets with 0
        targets_array[~num_position] = 0.0

        residual = (targets_array - outputs) * num_position_float_array
        residual = residual.reshape(residual.shape[0], -1)
        standardized_residual = np.dot(residual, torch.diag(feature_weights).numpy())
        average_standardized_residual = \
            np.sum(
                standardized_residual * standardized_residual,
                1
            ) \
            / \
            np.sum(
                num_position_float_array.reshape(
                    num_position_float_array.shape[0],
                    -1
                ),
                1
            )

        return average_standardized_residual


class BernoulliLoss(nn.Module):
    def __init__(self):
        super(BernoulliLoss, self).__init__()

    @staticmethod
    def link(output_float_variable):
        return 1.0 / (1.0 + torch.exp(-output_float_variable))

    def forward(self, output_float_variable, target_float_variable):
        outputs_after_link_float_variable = self.link(output_float_variable)
        result_float_variable = - torch.sum(
            (1.0 - target_float_variable)
            * torch.log(1.0 - outputs_after_link_float_variable)
            + target_float_variable * torch.log(outputs_after_link_float_variable)
        )
        return result_float_variable

    @staticmethod
    def hess(grad_long_double_array, exp_output_long_double_array, target_long_double_array):
        """
        Calculate Hessian diagonal line.
        :param grad_long_double_array:          a numpy array of the shape of
                                                [
                                                    batch_size * add_booster_layer_after_n_batch,
                                                    n_channel_from_boosting,
                                                    first_dim_from_boosting,
                                                    second_dim_from_boosting
                                                ]
        :param exp_output_long_double_array:    a numpy array of the shape of
                                                [
                                                    batch_size * add_booster_layer_after_n_batch,
                                                    1
                                                ]
        :param target_long_double_array:        a numpy array of the shape of
                                                [
                                                    batch_size * add_booster_layer_after_n_batch,
                                                    1
                                                ]
        :return:                                a numpy array of the shape of
                                                [
                                                    batch_size * add_booster_layer_after_n_batch,
                                                    n_channel_from_boosting,
                                                    first_dim_from_boosting,
                                                    second_dim_from_boosting
                                                ]
        """
        prediction_after_lf = exp_output_long_double_array / (1.0 + exp_output_long_double_array)
        denominator = target_long_double_array - prediction_after_lf
        hess = (
                   grad_long_double_array / denominator.reshape(-1, 1, 1, 1)
               ) ** 2.0 \
               * \
               (
                   prediction_after_lf
                   * (1.0 - prediction_after_lf)
               ).reshape(-1, 1, 1, 1)
        return hess

    @staticmethod
    def individual_loss(output_float_array, target_float_array):
        output_after_link_float_array = 1.0 / (1.0 + np.exp(-output_float_array))
        return - (
            (
                1.0 - target_float_array
            )
            * np.log(
                1.0 - output_after_link_float_array
            )
            + target_float_array
            * np.log(output_after_link_float_array)
        )
