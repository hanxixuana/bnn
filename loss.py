#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class FeatureNormalizedMSE(nn.Module):
    def __init__(self):
        super(FeatureNormalizedMSE, self).__init__()

    @staticmethod
    def torch_link(output, exp_exposure):
        return output

    @staticmethod
    def np_link(output, exp_exposure):
        return output

    def forward(self, output_float_variable, target_float_variable,
                e_exp_float_variable, sample_weight_float_variable, feature_weight_float_variable):
        """
        Loss.
        :param output_float_variable:           [batch_size, n_channel, n_first_dim, n_second_dim]
        :param target_float_variable:           [batch_size, n_channel, n_first_dim, n_second_dim]
        :param e_exp_float_variable:            [batch_size, 1]
        :param sample_weight_float_variable:    [batch_size, 1]
        :param feature_weight_float_variable:   [n_channel * n_first_dim * n_second_dim, 1]
        :return:                                Variable of one value
        """

        targets_array = target_float_variable.data.cpu().numpy()
        num_position = ~np.isnan(targets_array)

        # replace nan in targets with 0
        targets_array[~num_position] = 0.0

        if output_float_variable.is_cuda:
            target_float_variable = Variable(torch.from_numpy(targets_array).cuda())
            num_position_variable = Variable(torch.from_numpy(num_position.astype('float')).float().cuda())
        else:
            target_float_variable = Variable(torch.from_numpy(targets_array))
            num_position_variable = Variable(torch.from_numpy(num_position.astype('float')).float())

        residual = (target_float_variable - output_float_variable) * num_position_variable
        residual = residual.view(residual.size()[0], -1)

        standardized_residual = torch.mm(residual, torch.diag(feature_weight_float_variable[:, 0]))
        average_standardized_residual = \
            torch.sum(
                standardized_residual ** 2.0,
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

    def hess(self, grad_long_double_array, output_long_double_array, target_long_double_array,
             e_exp_long_double_array, weight_long_double_array):
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
        return np.ones_like(grad_long_double_array, dtype='float128')

    @staticmethod
    def individual_loss(output_array, target_array, feature_weight_array):
        """
        Nan-omitted feature-weighted individual MSE
        :param outputs: a 4-dimensional numpy array
        :param targets: a 4-dimensional numpy array
        :param feature_weights: a two-dimensional numpy array column vector
        :return: individual losses
        """
        targets_array = target_array.copy()
        num_position = ~np.isnan(targets_array)
        num_position_float_array = num_position.astype('float')

        # replace nan in targets with 0
        targets_array[~num_position] = 0.0

        residual = (targets_array - output_array) * num_position_float_array
        residual = residual.reshape(residual.shape[0], -1)
        standardized_residual = np.dot(residual, np.diag(feature_weight_array[:, 0]))
        average_standardized_residual = \
            np.sum(
                standardized_residual ** 2.0,
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

    def forward(self, output_float_variable, target_float_variable,
                e_exp_float_variable, sample_weight_float_variable, feature_weight_float_variable):

        output_after_link_float_variable = self.torch_link(output_float_variable, e_exp_float_variable)

        result_float_variable = - torch.sum(
            sample_weight_float_variable
            *
            (
                (1.0 - target_float_variable) * torch.log(1.0 - output_after_link_float_variable)
                +
                target_float_variable * torch.log(output_after_link_float_variable)
            )
        )

        return result_float_variable

    def hess(self, grad_long_double_array, output_long_double_array, target_long_double_array,
             e_exp_long_double_array, weight_long_double_array):
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

        output_float_vec = output_float_array[:, 0]
        target_float_vec = target_float_array[:, 0]

        if e_exp_float_array is None:
            e_exp_float_vec = np.ones_like(output_float_vec, dtype='float32')
        else:
            e_exp_float_vec = e_exp_float_array[:, 0]

        required_exposure = lift_at * np.sum(e_exp_float_vec)
        total_positive = np.sum(target_float_vec)

        sorted_indices_val_prediction = np.argsort(-output_float_vec)

        accumulated_exposure = 0.0
        accumulated_positive = 0.0
        pct = 0.0

        for item in sorted_indices_val_prediction.tolist():

            accumulated_positive += target_float_vec[item]
            accumulated_exposure += e_exp_float_vec[item]

            if accumulated_exposure > required_exposure:
                pct = accumulated_positive / total_positive
                break

        return pct
