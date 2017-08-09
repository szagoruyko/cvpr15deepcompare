import cv2
import os
import argparse
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser('DeepCompare data preparation')
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--test', action='store_true')


def read_matches_file(filename):
    print(filename)
    data = np.loadtxt(filename, dtype=np.uint64)
    mask = data[:,1] == data[:,4]
    pairs = data[:,(0, 3)]
    return pairs[mask], pairs[np.logical_not(mask)]


if __name__ == '__main__':
    opt = parser.parse_args()
    impath = opt.data_dir
    match_data = {name: read_matches_file(os.path.join(impath, name))
                  for name in os.listdir(impath) if name.startswith('m50_')}

    info = np.loadtxt(os.path.join(impath, 'info.txt'), dtype=np.uint64)[:,0]
    
    image_list = filter(lambda x: x.endswith('.bmp'), os.listdir(impath))

    patches = []
    for name in tqdm(sorted(image_list)):
        im = cv2.imread(os.path.join(impath, name), cv2.IMREAD_GRAYSCALE)
        patches.append(im.reshape(16,64,16,64).transpose(0,2,1,3).reshape(-1,64,64))
    patches = np.concatenate(patches)[:info.size]
    mean = np.mean(patches, axis=(1,2))

    np.save(arr={'patches': patches,
                 'mean': mean,
                 'info': info,
                 'match_data': match_data},
            file=open(os.path.join(impath, 'data.npy'), 'w'))

    if opt.test:
        import torch
        from torch.utils.serialization import load_lua
        data_torch = load_lua(os.path.join(impath, 'data.t7'))

        print((torch.from_numpy(patches).float() - data_torch['patches'].float()).abs().max())
