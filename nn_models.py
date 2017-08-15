#!/usr/bin/env python

import torch.nn as nn
import torch.nn.functional as F


class PureXGB(nn.Module):
    def __init__(self):
        super(PureXGB, self).__init__()

        self.n_channel_from_boosting = 1
        self.first_dim_from_boosting = 1
        self.second_dim_from_boosting = 1

        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        x = x.view(-1, 1)
        return x


class BoostFC(nn.Module):
    def __init__(self, n_channel_to_nn, first_dim_to_nn, second_dim_to_nn, dim_of_hidden_layer, dropout_prob=0.0):

        super(BoostFC, self).__init__()

        self.n_channel_from_boosting = n_channel_to_nn
        self.first_dim_from_boosting = first_dim_to_nn
        self.second_dim_from_boosting = second_dim_to_nn

        self.dropout_0 = nn.Dropout(dropout_prob)
        self.dropout_1 = nn.Dropout(dropout_prob)

        self.fc_0 = nn.Linear(
            self.n_channel_from_boosting
            * self.first_dim_from_boosting
            * self.second_dim_from_boosting,
            dim_of_hidden_layer
        )

        self.fc_1 = nn.Linear(
            dim_of_hidden_layer,
            1
        )

    def forward(self, x):

        x = x.view(-1,
                   self.n_channel_from_boosting *
                   self.first_dim_from_boosting *
                   self.second_dim_from_boosting)
        x = F.sigmoid(x)

        x = self.dropout_0(x)
        x = self.fc_0(x)
        x = F.elu(x)

        x = self.dropout_1(x)
        x = self.fc_1(x)

        return x


class Cov_nn(nn.Module):
    def __init__(self):
        super(Cov_nn, self).__init__()

        self.n_channel_from_boosting = 2
        self.first_dim_from_boosting = 8
        self.second_dim_from_boosting = 8

        self.features = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=4, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=1),
        )
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(8 * 2 * 2, 24),
            F.elu(),
            # nn.Dropout(),
            nn.Linear(24, 8),
            nn.Softmax()
        )

    # required
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 8 * 2 * 2)
        x = self.classifier(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, input_size, intermedia_size, dropout_prob=0.0):
        super(BasicBlock, self).__init__()
        self.input_size = input_size
        self.intermedia_size = intermedia_size

        self.bn1 = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, intermedia_size)
        self.bn2 = nn.BatchNorm1d(intermedia_size)
        self.fc2 = nn.Linear(intermedia_size, input_size)

        self.dropout = nn.Dropout(dropout_prob)
        self.dropout_prob = dropout_prob

    def forward(self, x):
        x = x.view(-1, self.input_size)
        residual = x
        out = x

        if self.dropout_prob > 0.0:
            out = self.dropout(out)
        out = self.bn1(out)
        out = F.elu(out)
        out = self.fc1(out)

        # if self.dropout_prob > 0.0:
        #     out = self.dropout(out)
        out = self.bn2(out)
        out = F.elu(out)
        out = self.fc2(out)

        out += residual
        return out


class ResRegNet(nn.Module):
    def __init__(self,
                 n_channel_from_boosting, first_dim_from_boosting, second_dim_from_boosting,
                 n_block, intermediate_size_in_block,
                 dropout_prob=0.5):
        """
        intermedia_size_in_block: array like, element[i] represents the size of
                                  intermedia node in i-th block
        """
        super(ResRegNet, self).__init__()

        self.dropout = nn.Dropout(dropout_prob)

        # Basic blocks, each block contains several channels
        self.blocks = nn.Sequential()
        for i in range(n_block):
            self.blocks.add_module(
                "block_%d" % (i),
                BasicBlock(
                    n_channel_from_boosting
                    * first_dim_from_boosting
                    * second_dim_from_boosting,
                    intermediate_size_in_block[i],
                    dropout_prob=dropout_prob
                )
            )

        self.fc = nn.Linear(
            n_channel_from_boosting * first_dim_from_boosting * second_dim_from_boosting,
            1
        )

        # save parameters for further use
        self.n_channel_from_boosting = n_channel_from_boosting
        self.first_dim_from_boosting = first_dim_from_boosting
        self.second_dim_from_boosting = second_dim_from_boosting

        self.n_block = n_block
        self.intermediate_size_in_block = intermediate_size_in_block

    def forward(self, x):

        x = self.dropout(x)

        x = x.view(-1,
                   self.n_channel_from_boosting
                   * self.first_dim_from_boosting
                   * self.second_dim_from_boosting)
        for block in self.blocks:
            x = block(x)

        x = self.fc(x)

        return x


class TwoLayerAE(nn.Module):
    def __init__(self, input_size, hidden_size_0, hidden_size_1):
        super(OneLayerAE, self).__init__()

        self.middle_layer = nn.Linear(hidden_size_0, hidden_size_1)
        self.output_layer = nn.Linear(hidden_size_1, input_size)

        self.n_channel_to_boosting = 1
        self.first_dim_to_boosting = 1
        self.second_dim_to_boosting = input_size

        self.n_channel_from_boosting = 1
        self.first_dim_from_boosting = 1
        self.second_dim_from_boosting = hidden_size_0

    def forward(self, x):

        x = x.view(-1,
                   self.n_channel_from_boosting
                   * self.first_dim_from_boosting
                   * self.second_dim_from_boosting
                   )
        x = F.elu(x)

        x = self.middle_layer(x)
        x = F.elu(x)

        x = self.output_layer(x)
        x = x.view(-1,
                   self.n_channel_to_boosting,
                   self.first_dim_to_boosting,
                   self.second_dim_to_boosting
                   )

        return x

class OneLayerAE(nn.Module):
    def __init__(self, input_size, hidden_size_0):
        super(OneLayerAE, self).__init__()

        self.output_layer = nn.Linear(hidden_size_0, input_size)

        self.n_channel_to_boosting = 1
        self.first_dim_to_boosting = 1
        self.second_dim_to_boosting = input_size

        self.n_channel_from_boosting = 1
        self.first_dim_from_boosting = 1
        self.second_dim_from_boosting = hidden_size_0

    def forward(self, x):

        x = x.view(-1,
                   self.n_channel_from_boosting
                   * self.first_dim_from_boosting
                   * self.second_dim_from_boosting
                   )
        x = F.elu(x)

        x = self.output_layer(x)
        x = x.view(-1,
                   self.n_channel_to_boosting,
                   self.first_dim_to_boosting,
                   self.second_dim_to_boosting
                   )

        return x