#!/usr/bin/env python
from __future__ import print_function

from copy import deepcopy
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import torchvision


def lift_plot_data(target, prediction, eexp, n_bucket):
    n_total_sample = prediction.shape[0]

    eexp_in_bucket = np.sum(eexp) / n_bucket

    weighted_exp_pred = np.exp(prediction) * eexp

    sorted_indices_val_prediction = np.argsort(-weighted_exp_pred / eexp)

    ave_response = []
    ave_weighted_exp_pred = []

    idx = 0

    for bucket_idx in range(n_bucket):

        accumulated_response = 0.0
        accumulated_weighted_exp_pred = 0.0
        accumulated_eexp = 0.0

        while accumulated_eexp <= eexp_in_bucket:

            find_idx = sorted_indices_val_prediction[idx]

            accumulated_response += target[find_idx]
            accumulated_weighted_exp_pred += weighted_exp_pred[find_idx]
            accumulated_eexp += eexp[find_idx]

            idx += 1
            if idx >= n_total_sample:
                break

        ave_response.append(accumulated_response / accumulated_eexp)
        ave_weighted_exp_pred.append(accumulated_weighted_exp_pred / accumulated_eexp)

    return np.array(ave_weighted_exp_pred[::-1]), np.array(ave_response[::-1])


def poisson_pct_obs_to_pct_of_response_sum(target, prediction, eexp, pct_of_sum):
    sum_all_response = np.sum(target)
    total_eexp = np.sum(eexp)

    sorted_indices_val_prediction = np.argsort(-np.exp(prediction))

    accumulated_sum = 0.0
    accumulated_eexp = 0.0
    pct_needed = 1.0

    for item in sorted_indices_val_prediction:
        accumulated_sum += target[item]
        accumulated_eexp += eexp[item]
        if accumulated_sum > pct_of_sum * sum_all_response:
            pct_needed = accumulated_eexp / total_eexp
            break

    return pct_needed


def bernoulli_pct_obs_to_pct_of_response_sum(target, prediction, pct_of_sum):
    sum_all_response = np.sum(target)
    total_count = target.shape[0]

    sorted_indices_val_prediction = np.argsort(-1.0 / (1.0 + np.exp(-prediction)))

    accumulated_sum = 0.0
    accumulated_count = 0.0
    pct_needed = 1.0

    for item in sorted_indices_val_prediction:
        accumulated_sum += target[item]
        accumulated_count += 1.0
        if accumulated_sum > pct_of_sum * sum_all_response:
            pct_needed = accumulated_count / total_count
            break

    return pct_needed


def bernoulli_lift_level(prediction, target, pct_of_sum, prob=False):
    sum_all_response = np.sum(target)
    total_count = float(target.shape[0])

    if not prob:
        sorted_indices_val_prediction = np.argsort(-1.0 / (1.0 + np.exp(-prediction)))
    else:
        sorted_indices_val_prediction = np.argsort(-prediction)

    accumulated_sum = 0.0
    accumulated_count = 0.0
    pct = 10.0

    for item in sorted_indices_val_prediction:
        accumulated_sum += float(target[item])
        accumulated_count += 1.0
        if accumulated_count > pct_of_sum * total_count:
            pct = accumulated_sum / sum_all_response
            break

    return pct


def get_data_set(path="data", download=False):
    train_set = torchvision.datasets.MNIST(path,
                                           train=True,
                                           download=download,
                                           transform=torchvision.transforms.ToTensor())
    train_data = train_set.train_data.numpy()
    train_labels = train_set.train_labels.numpy()

    val_set = torchvision.datasets.MNIST(path,
                                         train=False,
                                         download=download,
                                         transform=torchvision.transforms.ToTensor())
    val_data = val_set.test_data.numpy()
    val_labels = val_set.test_labels.numpy()

    return train_data, train_labels, val_data, val_labels


def get_data_set_loader(path, batch_size=1, num_workers=1, download=False, shuffle=False):
    if batch_size == -1:
        enable_full_set = True
        batch_size = 1
    else:
        enable_full_set = False

    train_set_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(path, train=True, download=download, transform=torchvision.transforms.ToTensor()),
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_set_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(path, train=False, download=download, transform=torchvision.transforms.ToTensor()),
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    if enable_full_set:
        batch_size = len(train_set_loader.dataset)
        train_set_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(path, train=True, download=download,
                                       transform=torchvision.transforms.ToTensor()),
            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        batch_size = len(test_set_loader.dataset)
        test_set_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(path, train=False, download=download,
                                       transform=torchvision.transforms.ToTensor()),
            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_set_loader, test_set_loader


def get_insurance_data(path='./data/insurance/processed_ins_data/', validate_proportion=0.3):
    """
    Get insurance data.
    :param path: the folder where the data is located
    :param validate_proportion: a proportion for validating with the rest for training
    :param random_seed: the seed to fix randomness
    :return: [n_sample, 35] numpy array for training
             [n_sample, 2] numpy array for training containing targets in the 1st column and eexp in the 2nd column
             [n_sample, 35] numpy array for validation
             [n_sample, 2] numpy array for validation containing targets in the 1st column and eexp in the 2nd column
    """

    features = pd.read_csv(path + 'features.csv', delimiter=',', header=None)
    targets = pd.read_csv(path + 'targets.csv', delimiter=',', header=None)
    eexp = pd.read_csv(path + 'eexp.csv', delimiter=',', header=None)

    n_sample = features.shape[0]
    validate_n_sample = int(n_sample * validate_proportion)

    shuffled_index = np.random.permutation(range(features.shape[0])).tolist()

    train_data = features.ix[shuffled_index[validate_n_sample:], :]
    train_target = targets.ix[shuffled_index[validate_n_sample:], :]
    train_eexp = eexp.ix[shuffled_index[validate_n_sample:], :]

    validate_data = features.ix[shuffled_index[:validate_n_sample], :]
    validate_target = targets.ix[shuffled_index[:validate_n_sample], :]
    validate_eexp = eexp.ix[shuffled_index[:validate_n_sample], :]

    return train_data.as_matrix(), \
           np.concatenate([train_target.as_matrix(), train_eexp.as_matrix()], axis=1), \
           validate_data.as_matrix(), \
           np.concatenate([validate_target.as_matrix(), validate_eexp.as_matrix()], axis=1)


def get_insurance_data_for_classification(path='./data/insurance/processed_ins_data/', validate_proportion=0.3):
    """
    Get insurance data.
    :param path: the folder where the data is located
    :param validate_proportion: a proportion for validating with the rest for training
    :param random_seed: the seed to fix randomness
    :return: [n_sample, 36] numpy array for training
             [n_sample, 1] numpy array for training containing targets in the 1st column
             [n_sample, 36] numpy array for validation
             [n_sample, 1] numpy array for validation containing targets in the 1st column
    """

    features = pd.read_csv(path + 'features.csv', delimiter=',', header=None)
    targets = pd.read_csv(path + 'targets.csv', delimiter=',', header=None)
    eexp = pd.read_csv(path + 'eexp.csv', delimiter=',', header=None)

    n_sample = features.shape[0]
    validate_n_sample = int(n_sample * validate_proportion)

    shuffled_index = np.random.permutation(range(features.shape[0])).tolist()

    train_data = features.ix[shuffled_index[validate_n_sample:], :]
    train_target = targets.ix[shuffled_index[validate_n_sample:], :]
    train_eexp = eexp.ix[shuffled_index[validate_n_sample:], :]

    validate_data = features.ix[shuffled_index[:validate_n_sample], :]
    validate_target = targets.ix[shuffled_index[:validate_n_sample], :]
    validate_eexp = eexp.ix[shuffled_index[:validate_n_sample], :]

    # ===
    return_train_data = np.concatenate(
        [
            train_data.as_matrix(),
            train_eexp.as_matrix()
        ],
        axis=1
    ).reshape(train_data.shape[0], 1, 1, -1).astype('float32')
    return_train_target = (train_target.as_matrix() > 0.5).astype('float32')
    return_val_data = np.concatenate(
        [
            validate_data.as_matrix(),
            validate_eexp.as_matrix()
        ],
        axis=1
    ).reshape(validate_data.shape[0], 1, 1, -1).astype('float32')
    return_val_target = (validate_target.as_matrix() > 0.5).astype('float32')

    return return_train_data, \
           return_train_target, \
           return_val_data, \
           return_val_target


def get_exp_ins_data_for_classification(path='./data/insurance/processed_ins_data/', validate_proportion=0.3):
    """
    Get insurance data.
    :param path: the folder where the data is located
    :param validate_proportion: a proportion for validating with the rest for training
    :param random_seed: the seed to fix randomness
    :return: [n_sample, 35] numpy array for training
             [n_sample, 1] numpy array for training containing targets in the 1st column
             [n_sample, 1] numpy array of exponential exposure
             [n_sample, 35] numpy array for validation
             [n_sample, 1] numpy array for validation containing targets in the 1st column
             [n_sample, 1] numpy array of exponential exposure

    """

    features = pd.read_csv(path + 'features.csv', delimiter=',', header=None)
    targets = pd.read_csv(path + 'targets.csv', delimiter=',', header=None)
    eexp = pd.read_csv(path + 'eexp.csv', delimiter=',', header=None)

    n_sample = features.shape[0]
    validate_n_sample = int(n_sample * validate_proportion)

    shuffled_index = np.random.permutation(range(n_sample)).tolist()

    # ===
    train_data = features.ix[shuffled_index[validate_n_sample:], :]
    train_target = targets.ix[shuffled_index[validate_n_sample:], :]
    train_eexp = eexp.ix[shuffled_index[validate_n_sample:], :]

    validate_data = features.ix[shuffled_index[:validate_n_sample], :]
    validate_target = targets.ix[shuffled_index[:validate_n_sample], :]
    validate_eexp = eexp.ix[shuffled_index[:validate_n_sample], :]

    # ===
    return_train_data = train_data.as_matrix().reshape(train_data.shape[0], 1, 1, -1).astype('float32')
    return_train_target = (train_target.as_matrix() > 0.5).astype('float32')
    return_train_e_exp = train_eexp.as_matrix().astype('float32')

    return_val_data = validate_data.as_matrix().reshape(validate_data.shape[0], 1, 1, -1).astype('float32')
    return_val_target = (validate_target.as_matrix() > 0.5).astype('float32')
    return_val_e_exp = validate_eexp.as_matrix().astype('float32')

    return return_train_data, \
           return_train_target, \
           return_train_e_exp, \
           return_val_data, \
           return_val_target, \
           return_val_e_exp


def get_bank_sales_data(path='./data/bank_sales_data/'):
    """
    Get insurance data.
    :param path: the folder where the data is located
    :param validate_proportion: a proportion for validating with the rest for training
    :param random_seed: the seed to fix randomness
    :return: [n_sample, 35] numpy array for training
             [n_sample, 2] numpy array for training containing targets in the 1st column and eexp in the 2nd column
             [n_sample, 35] numpy array for validation
             [n_sample, 2] numpy array for validation containing targets in the 1st column and eexp in the 2nd column
    """

    train_all = pd.read_csv(path + 'dummy_train.csv', delimiter=',')
    val_all = pd.read_csv(path + 'dummy_test.csv', delimiter=',')

    train_feature = train_all[train_all.columns[1:-1]].as_matrix().astype('float32')
    train_target = train_all[train_all.columns[-1]].fillna(0.0).as_matrix().astype('float32')

    val_feature = val_all[val_all.columns[1:-1]].as_matrix().astype('float32')
    val_target = val_all[val_all.columns[-1]].fillna(0.0).as_matrix().astype('float32')

    return train_feature, train_target, val_feature, val_target


def get_large_bank_sales_data(path='./data/large_bank_sales_data/', validate_proportion=1.0, random_seed=0):
    """
    Get insurance data.
    :param path: the folder where the data is located
    :param validate_proportion: a proportion for validating with the rest for training
    :param random_seed: the seed to fix randomness
    :return: [n_sample, 35] numpy array for training
             [n_sample, 2] numpy array for training containing targets in the 1st column and eexp in the 2nd column
             [n_sample, 35] numpy array for validation
             [n_sample, 2] numpy array for validation containing targets in the 1st column and eexp in the 2nd column
    """

    train = pd.read_csv(path + 'train_original_after_modification.csv', delimiter=',')
    test = pd.read_csv(path + 'test_original_after_modification.csv', delimiter=',')

    train_feature = train[train.columns[1:-3]].as_matrix().astype('float32')
    train_target = train[train.columns[-3]].fillna(0.0).as_matrix().astype('float32')[:, np.newaxis]

    # ======================

    n_sample = test.shape[0]
    validate_n_sample = int(n_sample * validate_proportion)

    np.random.seed(random_seed)
    shuffled_index = np.random.permutation(range(n_sample)).tolist()

    test_feature = test[test.columns[1:-3]].as_matrix().astype('float32')
    test_target = test[test.columns[-3]].fillna(0.0).as_matrix().astype('float32')[:, np.newaxis]

    val_feature = test_feature[shuffled_index[:validate_n_sample], :]
    val_target = test_target[shuffled_index[:validate_n_sample]]

    return train_feature, train_target, val_feature, val_target, test_feature, test_target


def get_two_number_from_mnist(two_numbers, path='./data/mnist/', validate_proportion=1.0):
    train_set = torchvision.datasets.MNIST(path, train=True)
    val_set = torchvision.datasets.MNIST(path, train=False)

    train_shape = train_set.train_data.size()
    val_shape = val_set.test_data.size()

    train_data = train_set.train_data.numpy().reshape(train_shape[0], 1, train_shape[1], train_shape[2])
    train_target = train_set.train_labels.numpy()

    train_two_numbers_idx_array = np.concatenate([np.where(train_target == item)[0] for item in two_numbers])
    train_two_numbers_idx_array = np.random.permutation(train_two_numbers_idx_array)

    train_data = train_data[train_two_numbers_idx_array].astype('float32')
    train_target = train_target[train_two_numbers_idx_array][:, np.newaxis].astype('float32')
    train_target[train_target == two_numbers[0]] = 0.0
    train_target[train_target == two_numbers[1]] = 1.0

    val_data = val_set.test_data.numpy().reshape(val_shape[0], 1, val_shape[1], val_shape[2])
    val_target = val_set.test_labels.numpy()

    val_two_numbers_idx_array = np.concatenate([np.where(val_target == item)[0] for item in two_numbers])
    val_two_numbers_idx_array = np.random.permutation(val_two_numbers_idx_array)

    val_data = val_data[val_two_numbers_idx_array].astype('float32')
    val_target = val_target[val_two_numbers_idx_array][:, np.newaxis].astype('float32')
    val_target[val_target == two_numbers[0]] = 0.0
    val_target[val_target == two_numbers[1]] = 1.0

    return train_data, train_target, val_data, val_target


def generate_param_list(xgb_param_cand, nn_param_cand, train_param_cand, default_param):

    n_xgb_cand = np.prod([xgb_param_cand[key].__len__() for key in xgb_param_cand if key != 'param_position'])
    n_nn_cand = np.prod([nn_param_cand[key].__len__() for key in nn_param_cand if key != 'param_position'])
    n_train_cand = np.prod([train_param_cand[key].__len__() for key in train_param_cand if key != 'param_position'])

    xgb_list = [deepcopy(default_param[xgb_param_cand['param_position']]) for _ in range(n_xgb_cand)]
    nn_list = [deepcopy(default_param[nn_param_cand['param_position']]) for _ in range(n_nn_cand)]
    train_list = [deepcopy(default_param[train_param_cand['param_position']]) for _ in range(n_train_cand)]

    # ==
    for idx in range(n_xgb_cand):
        temp_idx = idx
        for param in xgb_param_cand:
            if param != 'param_position':
                xgb_list[idx][param] = xgb_param_cand[param][temp_idx % len(xgb_param_cand[param])]
                temp_idx /= len(xgb_param_cand[param])

    for idx in range(n_nn_cand):
        temp_idx = idx
        for param in nn_param_cand:
            if param != 'param_position':
                nn_list[idx][param] = nn_param_cand[param][temp_idx % len(nn_param_cand[param])]
                temp_idx /= len(nn_param_cand[param])

    for idx in range(n_train_cand):
        temp_idx = idx
        for param in train_param_cand:
            if param != 'param_position':
                train_list[idx][param] = train_param_cand[param][temp_idx % len(train_param_cand[param])]
                temp_idx /= len(train_param_cand[param])

    param_list = []
    run_idx = 0
    for xgb_param in xgb_list:
        for nn_param in nn_list:
            for train_param in train_list:
                param_list.append((run_idx, deepcopy(xgb_param), deepcopy(nn_param), deepcopy(train_param)))
                run_idx += 1

    print('Generated %d sets of parameters.' % run_idx)

    return param_list


def param_optim(func_handle, param_list, n_process = 11):

    pool = Pool(n_process)

    pool.map(func_handle, param_list)






