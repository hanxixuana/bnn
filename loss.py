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
    def torch_link(output, exp_exposure):
        """
        :rtype: torch.FloatTensor
        """
        return 1.0 / (1.0 + torch.exp(-output) / exp_exposure)

    @staticmethod
    def np_link(output, exp_exposure):
        """
        :rtype: np.ndarray
        """
        return 1.0 / (1.0 + np.exp(-output) / exp_exposure)

    def forward(self, output_float_variable, target_float_variable, e_exp_float_variable, weight_float_variable):

        output_after_link_float_variable = self.torch_link(output_float_variable, e_exp_float_variable)

        result_float_variable = - torch.sum(
            weight_float_variable
            *
            (
                (1.0 - target_float_variable) * torch.log(1.0 - output_after_link_float_variable)
                +
                target_float_variable * torch.log(output_after_link_float_variable)
            )
        )

        return result_float_variable

    def hess(
            self,
            grad_long_double_array,
            output_long_double_array,
            target_long_double_array,
            e_exp_long_double_array,
            weight_long_double_array
    ):
        """
        Calculate Hessian diagonal line.

        :type grad_long_double_array:           np.ndarray
        :type output_long_double_array:         np.ndarray
        :type target_long_double_array:         np.ndarray
        :type e_exp_long_double_array:          np.ndarray
        :type weight_long_double_array:         np.ndarray

        :param grad_long_double_array:          a numpy array of the shape of
                                                [
                                                    batch_size * add_booster_layer_after_n_batch,
                                                    n_channel_from_boosting,
                                                    first_dim_from_boosting,
                                                    second_dim_from_boosting
                                                ]
        :param output_long_double_array:        a numpy array of the shape of
                                                [
                                                    batch_size * add_booster_layer_after_n_batch,
                                                    1
                                                ]
        :param target_long_double_array:        a numpy array of the shape of
                                                [
                                                    batch_size * add_booster_layer_after_n_batch,
                                                    1
                                                ]
        :param e_exp_long_double_array:           a numpy array of the shape of
                                                [
                                                    batch_size * add_booster_layer_after_n_batch,
                                                    1
                                                ]
        :param weight_long_double_array:        a numpy array of the shape of
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

        output_after_link_long_double_array = self.np_link(output_long_double_array, e_exp_long_double_array)

        denominator = output_after_link_long_double_array - target_long_double_array

        numerator = (
            output_after_link_long_double_array
            * (1.0 - output_after_link_long_double_array)
            / weight_long_double_array
        )

        coefficient = (numerator / denominator ** 2.0).reshape(-1, 1, 1, 1)

        hess = coefficient * grad_long_double_array ** 2.0

        return hess

    def lift_level(self, output_float_array, target_float_array, lift_at, e_exp_float_array=None):
        """
        Lift level at a certain value.
        :type output_float_array:   np.ndarray
        :type target_float_array:   np.ndarray
        :type lift_at:              float or double
        :type e_exp_float_array:    np.ndarray or None
        :rtype:                     float
        """

        output_float_array = output_float_array[:, 0]
        target_float_array = target_float_array[:, 0]

        if e_exp_float_array is None:
            e_exp_float_array = np.ones_like(output_float_array, dtype='float32')
        else:
            e_exp_float_array = e_exp_float_array[:, 0]

        output_after_link_float_array = self.np_link(output_float_array, e_exp_float_array)

        required_exposure = lift_at * np.sum(e_exp_float_array)
        total_positive = np.sum(target_float_array)

        sorted_indices_val_prediction = np.argsort(-output_after_link_float_array)

        accumulated_exposure = 0.0
        accumulated_positive = 0.0
        pct = 0.0

        for item in sorted_indices_val_prediction.tolist():

            accumulated_positive += target_float_array[item]
            accumulated_exposure += e_exp_float_array[item]

            if accumulated_exposure > required_exposure:
                pct = accumulated_positive / total_positive
                break

        return pct
