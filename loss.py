from abc import abstractmethod, ABCMeta
from copy import deepcopy

from torch import FloatTensor, diag, mm, sum, ones, zeros, exp, log, sort
from torch.autograd import Variable
from torch.nn import Module


class Loss(Module):

    __metaclass__ = ABCMeta

    def __init__(self):
        super(Loss, self).__init__()

    @abstractmethod
    def torch_link(self, output, exp_exposure):
        """
        Link function.
        :type output:          FloatTensor or Variable
        :type exp_exposure:    FloatTensor or Variable
        :rtype:                FloatTensor or Variable
        """
        raise NotImplementedError

    @abstractmethod
    def individual_loss(self, output, target, e_exp, sample_weight, feature_weight):
        """
        Samples' individual losses.
        :param FloatTensor or Variable output:          [batch_size, n_channel, n_first_dim, n_second_dim]
        :param FloatTensor or Variable target:          [batch_size, n_channel, n_first_dim, n_second_dim]
        :param FloatTensor or Variable e_exp:           [batch_size, 1]
        :param FloatTensor or Variable sample_weight:   [batch_size, 1]
        :param FloatTensor or Variable feature_weight:  [n_channel * n_first_dim * n_second_dim, 1]
        :return FloatTensor or Variable:                [batch_size, 1]
        """
        raise NotImplementedError

    def forward(self, output, target, e_exp, sample_weight, feature_weight):
        return sum(self.individual_loss(output, target, e_exp, sample_weight, feature_weight))

    @abstractmethod
    def hess(self, grad, output, target, e_exp, sample_weight):
        """
        Calculate Hessian diagonal line.
        :param FloatTensor grad:            [batch_size, n_channel, first_dim, second_dim]
        :param FloatTensor output:          [batch_size, 1]
        :param FloatTensor target:          [batch_size, 1]
        :param FloatTensor e_exp:           [batch_size, 1]
        :param FloatTensor sample_weight:   [batch_size, 1]
        :return FloatTensor:                [batch_size, n_channel, first_dim, second_dim]
        """
        raise NotImplementedError


class FeatureNormalizedMSE(Loss):
    def __init__(self):
        super(FeatureNormalizedMSE, self).__init__()

    def torch_link(self, output, exp_exposure):
        return output

    def individual_loss(self, output, target, e_exp, sample_weight, feature_weight):

        sample_n = target.size()[0]

        target_without_nan = deepcopy(target)
        target_without_nan[target != target] = 0.0

        num_position = (target == target).float()

        residual = (target_without_nan - self.torch_link(output, e_exp)) * num_position

        standardized_residual = mm(
            residual.view(sample_n, -1),
            diag(feature_weight[:, 0])
        )

        average_standardized_residual = (
            sample_weight
            *
            sum(
                standardized_residual ** 2.0,
                1
            )
            /
            sum(
                num_position.view(
                    sample_n,
                    -1
                ),
                1
            )
        )

        return average_standardized_residual

    def hess(self, grad, output, target, e_exp, sample_weight):
        if grad.is_cuda:
            return ones(grad.size()).cuda()
        else:
            return ones(grad.size())


class BernoulliLoss(Loss):

    numerical_cap = 16.0
    numerical_floor = -16.0

    def __init__(self):
        super(BernoulliLoss, self).__init__()

    def torch_link(self, output, exp_exposure):
        output = self.correct_output(output)
        return 1.0 / (1.0 + exp(-output) / exp_exposure)

    def correct_output(self, output):

        if output.is_cuda:
            correction = zeros(output.size()).cuda()
        else:
            correction = zeros(output.size())

        if isinstance(output, Variable):

            too_large_pos = output.data > self.numerical_cap
            too_small_pos = output.data < self.numerical_floor

            correction[too_large_pos] = self.numerical_cap - output.data[too_large_pos]
            correction[too_small_pos] = self.numerical_floor - output.data[too_small_pos]

            correction = Variable(correction, requires_grad=False)

        else:

            too_large_pos = output > self.numerical_cap
            too_small_pos = output < self.numerical_floor

            correction[too_large_pos] = self.numerical_cap - output[too_large_pos]
            correction[too_small_pos] = self.numerical_floor - output[too_small_pos]

        output += correction

        return output

    def individual_loss(self, output, target, e_exp, sample_weight, feature_weight):

        output_after_link = self.torch_link(output, e_exp)

        result_float_variable = (
            - sample_weight
            *
            (
                (1.0 - target) * log(1.0 - output_after_link)
                +
                target * log(output_after_link)
            )
        )

        return result_float_variable

    def hess(self, grad, output, target, e_exp, sample_weight):

        output_after_link = self.torch_link(output, e_exp)

        denominator = output_after_link - target

        numerator = (
            output_after_link
            * (1.0 - output_after_link)
            / sample_weight
        )

        coefficient = (numerator / denominator / denominator).unsqueeze(2).unsqueeze(3).expand(grad.size())

        hess = coefficient * grad ** 2.0

        return hess

    @staticmethod
    def lift_level(output, target, lift_at, e_exp=None):
        """
        Lift level at a certain value.
        :param FloatTensor output:          [n_sample, 1]
        :param FloatTensor target:          [n_sample, 1]
        :param float or double lift_at:     value
        :param FloatTensor or None e_exp:   [n_sample, 1]
        :return float:                      value
        """
        output_vec = output[:, 0]
        target_vec = target[:, 0]

        if e_exp is None:
            if output.is_cuda:
                e_exp_vec = ones(output_vec.size()).cuda()
            else:
                e_exp_vec = ones(output_vec.size())
        else:
            e_exp_vec = e_exp[:, 0]

        required_exposure = lift_at * sum(e_exp_vec)
        total_positive = sum(target_vec)

        _, sorted_indices_val_prediction = sort(-output_vec)

        accumulated_exposure = 0.0
        accumulated_positive = 0.0
        pct = 0.0

        for item in sorted_indices_val_prediction:

            accumulated_positive += target_vec[item]
            accumulated_exposure += e_exp_vec[item]

            if accumulated_exposure > required_exposure:
                pct = accumulated_positive / total_positive
                break

        return pct
