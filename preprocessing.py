
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy import signal
import argparse

MAT_COLUMNS = ["export_X", "export_Y", "export_Type", "export_Trial", "export_OnPaper", "subject_num"]
EMG_COLUMNS = ['EMG_0', 'EMG_1', 'EMG_2', 'EMG_3', 
               'EMG_4', 'EMG_5', 'EMG_6', 'EMG_7']

def parse_args():
    parser = argparse.ArgumentParser('EMG Preprocessing script')
    parser.add_argument('-d', '--data_path', required=True,
                        help="Path to all data stored in .npz format (after converter)")
    parser.add_argument('-n', '--name', required=True,
                        help="Result file name with .csv format")
    parser.add_argument('-bo', '--butter_order', required=False,
                        default=2, type=int,
                        help="Butterworth filter order")
    parser.add_argument('-cf', '--c_freq', required=False,
                        default=1, type=int,
                        help="Frequency cutoff of filter")
    parser.add_argument('-sf', '--s_freq', required=False,
                        default=1000, type=int,
                        help="Sampling frequency")
    parser.add_argument('-ss', '--sparse_step', required=False,
                        default=20, type=int,
                        help="Number of elements in one trial")
    return parser.parse_args()

def butter_processing(work_data: pd.DataFrame, 
                      fs: int, fc: int, butter_order: int) -> pd.DataFrame:
    """Butterworth filer processing

    Args:
        work_data (DataFrame): All subjects EMGS data
        fs (int): Sampling Frequencey
        fc (int): Cutoff Frequency
        butter_order (int): Butterworth order

    Returns:
        DataFrame: preprocessed data
    """
    w = (fc * 2) / fs            # Normalize the frequency
    b, a = signal.butter(butter_order, w, 'low')
    
    lst = []
            
    for emg_col in tqdm(EMG_COLUMNS):
        tmp_data = work_data[emg_col].values
        tmp_data = (tmp_data - tmp_data.mean())/np.std(tmp_data)
        tmp_data = np.abs(tmp_data)
        tmp_data = signal.filtfilt(b, a, tmp_data)
        
        lst.append(tmp_data)

    for col in ["export_X", "export_Y", "export_Type", "export_Trial", "export_OnPaper", "subject_num"]:
        lst.append(work_data[col].values)

    work_data= pd.DataFrame(np.array(lst).T, columns=work_data.columns)

    return work_data


def on_paper_export(work_data):
    lst = []
    mask = work_data["export_OnPaper"].values.astype('bool')
    for emg_col in EMG_COLUMNS:
        tmp_data = work_data[emg_col].values
        tmp_data = tmp_data[mask]
        
        lst.append(tmp_data)

    for col in ["export_X", "export_Y", "export_Type", "export_Trial", "subject_num"]:
        lst.append(work_data[col].values[mask])
        
    columns = list(work_data.columns)
    columns.remove("export_OnPaper")
    work_data_sparse = pd.DataFrame(np.array(lst).T, columns=columns)

    return work_data_sparse


def scale_data(data):
    """

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    trials = np.unique(data["export_Trial"])

    handled_data = [[], []]
    
    for trial in trials:
        curr_data = data[data["export_Trial"] == trial]
        for i, col in enumerate(["export_X", "export_Y"]):
            tmp = curr_data[col].values
            min_v = tmp.min()
            max_v = tmp.max()
            tmp = tmp - max_v + (max_v - min_v)/2
            tmp -= tmp.min()
            tmp /= tmp.max()
            handled_data[i].extend(tmp)
        print("Trial {}".format(round(trial / max(trials), 2)), end='\r')
    for i, col in enumerate(["export_X", "export_Y"]):
        data[col] = handled_data[i]
    return data


def reconstruct(path):
    data = np.load(args.data_path)['arr_0']
    print(data.shape)
    df = pd.DataFrame(data, columns=EMG_COLUMNS + MAT_COLUMNS)

    return df

def main(path, fs, fc, butter_order):
    df = reconstruct(path)
    df = butter_processing(df, fs=args.s_freq, fc=args.c_freq, 
                           butter_order=args.butter_order)
    df = on_paper_export(df)
    print("Saving data...")
    np.savez_compressed(args.name, df.values)

if __name__ == "__main__":
    args = parse_args()
    df = main(args.data_path, fs=args.s_freq, fc=args.c_freq, 
                           butter_order=args.butter_order)