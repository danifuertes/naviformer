import os
import pickle
import numpy as np


def parse_softmax_temp(raw_temp):
    if os.path.isfile(raw_temp):
        return np.loadtxt(raw_temp)[-1, 0]
    return float(raw_temp)


def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def load_dataset(filename):
    with open(check_extension(filename), 'rb') as f:
        return pickle.load(f)


def save_dataset(dataset, filename):

    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def inputs2device(batch, device):
    if isinstance(batch, dict):
        for k, v in batch.items():
            batch[k] = v.unsqueeze(0).to(device)
    else:
        batch = batch.unsqueeze(0).to(device)
    return batch


def batch2numpy(batch, to_list=False):
    if isinstance(batch, dict):
        for k, v in batch.items():
            batch[k] = v.cpu().detach().numpy()
            batch[k] = batch[k].tolist() if to_list else batch[k].squeeze()
    else:
        batch = batch.cpu().detach().numpy()
        batch = batch.tolist() if to_list else batch.squeeze()
    return batch


def actions2numpy(actions, end_ids):
    if isinstance(actions, list):
        out = []
        for i, action in enumerate(actions):
            out.append(action.cpu().detach().numpy().squeeze(0))
            if out[-1][0] != 0:
                out[-1] = np.concatenate(([0], out[-1]), axis=-1)
            elif out[-1][-1] != end_ids:
                out[-1] = np.concatenate((out[-1], [end_ids]), axis=-1)
    else:
        out = actions.cpu().detach().numpy().squeeze(0)
    return out
