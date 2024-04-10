import numpy as np
from tqdm import tqdm
from random import random
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def train_test_df_split(df, step):
    trials = np.unique(df["export_Trial"])

    train_df = df.copy()

    remove_train_trials = trials[1::step]
    remove_test_trials = [trial for trial in trials if trial not in remove_train_trials]

    for trial in remove_train_trials:
        train_df = train_df.drop((train_df[train_df["export_Trial"] == trial]).index)

    test_df = df.copy()
    for trial in remove_test_trials:
        test_df = test_df.drop((test_df[test_df["export_Trial"] == trial]).index)
    return train_df, test_df

def window_maker(data, T, k, tpe):
    trials = np.unique(data["export_Trial"])
    windows = np.array([sliding_window_view(
                            np.pad(
                                data[data["export_Trial"] == trial].values[:, :8],
                                ((T//2, T//2), (0, 0)),
                                tpe
                            ), (T, k)
                        )[:, 0]
                        for trial in trials])
    windows = windows.reshape(-1, windows.shape[2], windows.shape[3])   

    return windows

def train_test_handling(train_df, test_df, win=False, win_shape=None, tpe=None):
    train_reg_data = train_df.values[:, :10]
    y_train = train_reg_data[:, 8:]
    train_trials = train_df["export_Trial"].values
    train_type = train_df["export_Type"].values
    
    test_reg_data = test_df.values[:, :10]
    y_test = test_reg_data[:, 8:]
    test_trials = test_df["export_Trial"].values
    test_type = test_df["export_Type"].values
    
    if win:
        X_train = window_maker(train_df, win_shape[0], win_shape[1], tpe)
        X_train = X_train.reshape(X_train.shape[0], -1)
        
        X_test = window_maker(test_df, win_shape[0], win_shape[1], tpe)
        X_test = X_test.reshape(X_test.shape[0], -1)
    else:
        X_train = train_reg_data[:, :8]
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = test_reg_data[:, :8]
        X_test = X_test.reshape(X_test.shape[0], -1)

    return X_train, y_train, train_trials, train_type, \
           X_test, y_test, test_trials, test_type

def emg_augmentation(df, n=10, p=0.5, d=4):
    def get_mask(mask, n):
        return np.repeat(mask[:, None], n, 1).astype('bool')


    def create_mask(data, value):
        mask = data == value
        return get_mask(mask, 8)

    values = df.values[:, :8]
    trials = df["export_Trial"].values
    types = df["export_Type"].values

    curr_add_trial = max(np.unique(trials)) + 1
    subject_num = 512
    for _ in tqdm(range(n)):
        for num in range(1, 11):
            mask = types == num
            mask_full = get_mask(mask, 8)
            general_values = values[mask_full].reshape(-1, 8)
            general_trial = trials[mask]
            num_trials = np.unique(general_trial)
            for trial in num_trials:
                gen_mask = create_mask(general_trial, trial)
                curr_data = general_values[gen_mask].reshape(-1, 8)
                if random() < p:
                    switch_trial = np.random.choice(num_trials)
                    indices = np.random.permutation(list(range(8)))
                    switch_emg_choice = indices[:d] 
                    original_emg_choice = indices[d:]
                    switch_data = general_values[create_mask(general_trial, switch_trial)].reshape(-1, 8)[:, switch_emg_choice]
                    original_data = general_values[gen_mask].reshape(-1, 8)[:, original_emg_choice]
                    result_data = general_values[gen_mask].reshape(-1, 8)
                    result_data[:, switch_emg_choice] = switch_data
                    result_data[:, original_emg_choice] = original_data
                    switch_data = np.concatenate((result_data, df[df["export_Trial"] == trial].values[:, 8:11]), axis=1)
                    switch_data = np.concatenate((switch_data, curr_add_trial*np.ones((len(curr_data), 1))), axis=1)
                    switch_data = np.concatenate((switch_data, subject_num*np.ones((len(curr_data), 1))), axis=1)
                    curr_add_trial += 1
                    df = pd.concat((df, pd.DataFrame(data=switch_data, columns=df.columns)))
    return df