import numpy as np
from tqdm import tqdm
import glob
import scipy.io
import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser('EMG Preprocessing script')
    parser.add_argument('-d', '--data_path', required=True,
                        help="Path to data folder with mat files")
    parser.add_argument('-n', '--name', required=True,
                        help="Result file name with .csv format")
    return parser.parse_args()


def convert_data(path, name):
    paths = sorted(glob.glob('data/*'))
    assert paths 
    mat_columns = ["export_X", "export_Y", "export_Type", "export_Trial", "export_OnPaper"]
    emg_columns = ['EMG_0', 'EMG_1', 'EMG_2', 'EMG_3', 
                'EMG_4', 'EMG_5', 'EMG_6', 'EMG_7']

    main_df = pd.DataFrame(columns=emg_columns+mat_columns)

    subject_nums = []

    for ii, path in tqdm(enumerate(paths)):
        mat = scipy.io.loadmat(path)
        mask = np.array(mat['export_OnPaper'].reshape(-1)).astype('bool')
        data = np.zeros((mask.shape[0], len(emg_columns) + len(mat_columns)))
        emg_data = np.array(mat["export_EMG"]) \
                    .astype('float32') \
                    .reshape(-1, 8)
        data[:, :len(emg_columns)] = emg_data
        
        for i, col in enumerate(mat_columns):
            tmp_data = np.array(mat[col].reshape(-1)).astype('float32')

            data[:, i + len(emg_columns)] = tmp_data
        tmp_df = pd.DataFrame(columns=main_df.columns, data=data)
        # main_df = main_df.append(tmp_df, ignore_index = True)
        main_df = pd.concat([main_df, tmp_df])

        subject_nums.extend([ii]*data.shape[0])
    main_df['subject_num'] = subject_nums

    np.savez_compressed(name, main_df.values)


if __name__ == "__main__":
    args = parse_args()
    convert_data(args.data_path, args.name)