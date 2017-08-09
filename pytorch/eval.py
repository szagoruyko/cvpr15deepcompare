from __future__ import print_function
import os
import sys
import argparse
from functools import partial
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.serialization import load_lua
from torchnet.dataset import ListDataset, ConcatDataset
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn import metrics
from scipy import interpolate
from torch.backends import cudnn
cudnn.benchmark = True

parser = argparse.ArgumentParser(description='DeepCompare PyTorch evaluation code')

parser.add_argument('--model', default='2ch', type=str)
parser.add_argument('--lua_model', default='', type=str, required=True)
parser.add_argument('--nthread', default=4, type=int)
parser.add_argument('--gpu_id', default='0', type=str)

parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--test_set', default='liberty', type=str)
parser.add_argument('--test_matches', default='m50_100000_100000_0.txt', type=str)


def get_iterator(dataset, batch_size, nthread):
    def get_list_dataset(pair_type):
        ds = ListDataset(elem_list=dataset[pair_type],
                         load=lambda idx: {'input': np.stack((dataset['patches'][v].astype(np.float32)
                                                              - dataset['mean'][v]) / 256.0 for v in idx),
                                           'target': 1 if pair_type == 'matches' else -1})
        ds = ds.transform({'input': torch.from_numpy, 'target': lambda x: torch.LongTensor([x])})

        return ds.batch(policy='include-last', batchsize=batch_size // 2)

    concat = ConcatDataset([get_list_dataset('matches'),
                            get_list_dataset('nonmatches')])

    return concat.parallel(batch_size=2, shuffle=False, num_workers=nthread)


def conv2d(input, params, base, stride=1, padding=0):
    return F.conv2d(input, params[base + '.weight'], params[base + '.bias'],
                    stride, padding)


def linear(input, params, base):
    return F.linear(input, params[base + '.weight'], params[base + '.bias'])


#####################   2ch   #####################

def deepcompare_2ch(input, params):
    o = conv2d(input, params, 'conv0', stride=3)
    o = F.max_pool2d(F.relu(o), 2, 2)
    o = conv2d(o, params, 'conv1')
    o = F.max_pool2d(F.relu(o), 2, 2)
    o = conv2d(o, params, 'conv2')
    o = F.relu(o).view(o.size(0), -1)
    return linear(o, params, 'fc')


#####################   2ch2stream   #####################

def deepcompare_2ch2stream(input, params):

    def stream(input, name):
        o = conv2d(input, params, name + '.conv0')
        o = F.max_pool2d(F.relu(o), 2, 2)
        o = conv2d(o, params, name + '.conv1')
        o = F.max_pool2d(F.relu(o), 2, 2)
        o = conv2d(o, params, name + '.conv2')
        o = F.relu(o)
        o = conv2d(o, params, name + '.conv3')
        o = F.relu(o)
        return o.view(o.size(0), -1)

    o_fovea = stream(F.avg_pool2d(input, 2, 2), 'fovea')
    o_retina = stream(F.pad(input, (-16,) * 4), 'retina')
    o = linear(torch.cat([o_fovea, o_retina], dim=1), params, 'fc0')
    return linear(F.relu(o), params, 'fc1')


#####################   siam   #####################

def siam(patch, params):
    o = conv2d(patch, params, 'conv0', stride=3)
    o = F.max_pool2d(F.relu(o), 2, 2)
    o = conv2d(o, params, 'conv1')
    o = F.max_pool2d(F.relu(o), 2, 2)
    o = conv2d(o, params, 'conv2')
    o = F.relu(o)
    return o.view(o.size(0), -1)


def deepcompare_siam(input, params):
    o = linear(torch.cat(map(partial(siam, params=params), input.split(1, dim=1)),
                         dim=1), params, 'fc0')
    return linear(F.relu(o), params, 'fc1')


def deepcompare_siam_l2(input, params):
    def single(patch):
        return F.normalize(siam(patch, params))
    return - F.pairwise_distance(*map(single, input.split(1, dim=1)))


#####################   siam2stream   #####################


def siam_stream(patch, params, base):
    o = conv2d(patch, params, base + '.conv0', stride=2)
    o = F.max_pool2d(F.relu(o), 2, 2)
    o = conv2d(o, params, base + '.conv1')
    o = F.relu(o)
    o = conv2d(o, params, base + '.conv2')
    o = F.relu(o)
    o = conv2d(o, params, base + '.conv3')
    return o.view(o.size(0), -1)


def streams(patch, params):
    o_retina = siam_stream(F.pad(patch, (-16,) * 4), params, 'retina')
    o_fovea = siam_stream(F.avg_pool2d(patch, 2, 2), params, 'fovea')
    return torch.cat([o_retina, o_fovea], dim=1)


def deepcompare_siam2stream(input, params):
    embeddings = map(partial(streams, params=params), input.split(1, dim=1))
    o = linear(torch.cat(embeddings, dim=1), params, 'fc0')
    o = F.relu(o)
    o = linear(o, params, 'fc1')
    return o


def deepcompare_siam2stream_l2(input, params):
    def single(patch):
        return F.normalize(streams(patch, params))
    return - F.pairwise_distance(*map(single, input.split(1, dim=1)))


models = {
    '2ch': deepcompare_2ch,
    '2ch2stream': deepcompare_2ch2stream,
    'siam': deepcompare_siam,
    'siam_l2': deepcompare_siam_l2,
    'siam2stream': deepcompare_siam2stream,
    'siam2stream_l2': deepcompare_siam2stream_l2,
}


def main(args):
    opt = parser.parse_args(args)
    print('parsed options:', vars(opt))

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    if torch.cuda.is_available():
        # to prevent opencv from initializing CUDA in workers
        torch.randn(8).cuda()
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    def load_provider():
        print('Loading test data')

        p = np.load(opt.test_set)[()]

        for i, t in enumerate(['matches', 'nonmatches']):
            p[t] = p['match_data'][opt.test_matches][i]

        return p

    test_iter = get_iterator(load_provider(), opt.batch_size, opt.nthread)

    def cast(t):
        return t.cuda() if torch.cuda.is_available() else t

    f = models[opt.model]
    net = load_lua(opt.lua_model)

    if opt.model == '2ch':
        params = {}
        for j, i in enumerate([0, 3, 6]):
            params['conv%d.weight' % j] = net.get(i).weight
            params['conv%d.bias' % j] = net.get(i).bias
        params['fc.weight'] = net.get(9).weight
        params['fc.bias'] = net.get(9).bias
    elif opt.model == '2ch2stream':
        params = {}
        for j, branch in enumerate(['fovea', 'retina']):
            for k, layer in enumerate(map(net.get(0).get(j).get(1).get, [1, 4, 7, 9])):
                params['%s.conv%d.weight' % (branch, k)] = layer.weight
                params['%s.conv%d.bias' % (branch, k)] = layer.bias
        for k, layer in enumerate(map(net.get, [1, 3])):
            params['fc%d.weight' % k] = layer.weight
            params['fc%d.bias' % k] = layer.bias
    elif opt.model == 'siam' or opt.model == 'siam_l2':
        params = {}
        for k, layer in enumerate(map(net.get(0).get(0).get, [1, 4, 7])):
            params['conv%d.weight' % k] = layer.weight
            params['conv%d.bias' % k] = layer.bias
        for k, layer in enumerate(map(net.get, [1, 3])):
            params['fc%d.weight' % k] = layer.weight
            params['fc%d.bias' % k] = layer.bias
    elif opt.model == 'siam2stream' or opt.model == 'siam2stream_l2':
        params = {}
        for stream, name in zip(net.get(0).get(0).modules, ['retina', 'fovea']):
            for k, layer in enumerate(map(stream.get, [2, 5, 7, 9])):
                params['%s.conv%d.weight' % (name, k)] = layer.weight
                params['%s.conv%d.bias' % (name, k)] = layer.bias
        for k, layer in enumerate(map(net.get, [1, 3])):
            params['fc%d.weight' % k] = layer.weight
            params['fc%d.bias' % k] = layer.bias

    params = {k: Variable(cast(v)) for k, v in params.items()}

    def create_variables(sample):
        inputs = Variable(cast(sample['input'].float().view(-1, 2, 64, 64)))
        targets = Variable(cast(sample['target'].float().view(-1)))
        return inputs, targets

    test_outputs, test_targets = [], []
    for sample in tqdm(test_iter, dynamic_ncols=True):
        inputs, targets = create_variables(sample)
        y = f(inputs, params)
        test_targets.append(sample['target'].view(-1))
        test_outputs.append(y.data.cpu().view(-1))

    fpr, tpr, thresholds = metrics.roc_curve(torch.cat(test_targets).numpy(),
                                             torch.cat(test_outputs).numpy(), pos_label=1)
    fpr95 = float(interpolate.interp1d(tpr, fpr)(0.95))

    print('FPR95:', fpr95)

    return fpr95


if __name__ == '__main__':
    main(sys.argv[1:])
