import __init__
from __init__ import *
import matplotlib
# matplotlib.use("agg")
import matplotlib.pyplot as plt
import Meta
import json
import torch
import math
import copy

from aux import _load_features_of_speakers, JsonDataset

nlyr = Meta.model_meta['attend_layers'] + 1
en = Meta.model_meta['keepsize_num_en']
de = Meta.model_meta['keepsize_num_de']
if de == en: de = 'en'

all_speakers = Meta.data_meta['speakers']
pair = Meta.data_meta['pairing'][0]
two_speakers = [all_speakers[pair[0]], all_speakers[pair[1]]]

# TODO
cumu = False
topdown_scan = True
Type = ['spec', 'spec_ami'][1]

BS = 20

"""
topdown=True:
000 | 000
001 | 001
011 | 010
111 | 100
topdown=False:
000 | 000
100 | 100
110 | 010
111 | 001
"""

def attentions_and_probe():
    conditions = []
    n_attentions = len(Meta.model_meta['layer_attentions'])

    layer_attentions = [0 for _ in range(n_attentions)]
    conditions.append(copy.copy(layer_attentions))
    for iatt in range(n_attentions):
        if topdown_scan:
            layer_attentions[n_attentions - 1 - iatt] = 1
            if cumu == False and iatt > 0:  # cancel previous layer attention
                layer_attentions[n_attentions - iatt] = 0
        else:
            layer_attentions[iatt] = 1
            if cumu == False and iatt > 0: # cancel previous layer attention
                layer_attentions[iatt - 1] = 0 
        
        conditions.append(copy.copy(layer_attentions))

    return conditions


def _make_FloatTensor(array):
    if torch.cuda.is_available(): return torch.cuda.FloatTensor(array,device=Meta.DEVICE_ID)
    else:                         return torch.Tensor(array)


def Corr(a,b):
    a = a - np.mean(a)
    b = b - np.mean(b)
    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum())
    return r


def AMI(sp1, sp2, re1, re2):
    return Corr(sp1, re1) + Corr(sp2, re2) - Corr(sp1, re2) - Corr(sp2, re1)


def do_results_spec(aNet, aeNet):
    dataset1 = JsonDataset(two_speakers[0], train=False, do_shuffle=False)
    dataset2 = JsonDataset(two_speakers[1], train=False, do_shuffle=False)
    N = len(dataset1)
    batches = N // BS

    features = _load_features_of_speakers()

    results = [[], [], [], []]

    for ibatch in range(batches):
        print(f"ibatch = {ibatch}")
        indices = range(ibatch * BS, (ibatch+1) * BS)
        sp1_batch = spectrogram([dataset1[i] for i in indices]) # BS x slice_len
        sp2_batch = spectrogram([dataset2[i] for i in indices]) # BS x slice_len
        mix_batch = spectrogram([dataset1[i] + dataset2[i] for i in indices]) # BS x slice_len
        mix_batch = _make_FloatTensor(mix_batch)

        batch_results = []
        for sp in two_speakers:
            attentions = aNet.forward(\
                        torch.cat([features[sp].view(1, 256) for _ in range(BS)], dim=0))
            spec = aeNet.forward(mix_batch, attentions).view(-1, *Meta.data_meta['specgram_size']) # BS x *spec_size

            batch_results.append(spec.cpu().detach().numpy())

        results[0].append(batch_results[0])
        results[1].append(batch_results[1])
        results[2].append(sp1_batch)
        results[3].append(sp2_batch)

    p1 = np.concatenate(results[0], axis=0).tolist()
    p2 = np.concatenate(results[1], axis=0).tolist()
    p3 = np.concatenate(results[2], axis=0).tolist()
    p4 = np.concatenate(results[3], axis=0).tolist()

    return [p1, p2, p3, p4]


def do_results_spec_ami(aNet, aeNet):
    dataset1 = JsonDataset(two_speakers[0], train=False, do_shuffle=False)
    dataset2 = JsonDataset(two_speakers[1], train=False, do_shuffle=False)
    N = len(dataset1)
    batches = N // BS

    features = _load_features_of_speakers()

    results = []

    for ibatch in range(batches):
        print(f"ibatch = {ibatch}")
        indices = range(ibatch * BS, (ibatch+1) * BS)
        sp1_batch = spectrogram([dataset1[i] for i in indices]) # BS x slice_len
        sp2_batch = spectrogram([dataset2[i] for i in indices]) # BS x slice_len
        mix_batch = spectrogram([dataset1[i] + dataset2[i] for i in indices]) # BS x slice_len
        mix_batch = _make_FloatTensor(mix_batch)

        batch_results = []
        for sp in two_speakers:
            attentions = aNet.forward(\
                        torch.cat([features[sp].view(1, 256) for _ in range(BS)], dim=0))
            spec = aeNet.forward(mix_batch, attentions).cpu().detach().numpy()

            batch_results.append(spec)

        batch_results = [AMI(sp1_batch[i], sp2_batch[i], batch_results[0][i], batch_results[1][i]) for i in range(BS)]
        results.append(batch_results)

    results = np.concatenate(results, axis=0).tolist()
    return results


def do_dump(aNet, aeNet, conditions):
    dump_dir = os.path.join(os.path.dirname(__file__) , f"../results/dumped/n={nlyr}_en={en}_resblock_3x3conv_5shortcut/de={de}/")
    cumu_str = {True:"cumu", False:"single"}[cumu]
    topdown_str = {True:"tpdwn", False:"botup"}[topdown_scan]
    if not os.path.isdir(dump_dir):
        os.makedirs(dump_dir)

    for icondition, condition in enumerate(conditions):
        Meta.model_meta['layer_attentions'] = condition
        if Type == 'spec':
            file_name = f"condition_{cumu_str}-{topdown_str}_{icondition}_spec.json"
            results = do_results_spec(aNet, aeNet)
        else:
            file_name = f"condition_{cumu_str}-{topdown_str}_{icondition}_spec_ami.json"
            results = do_results_spec_ami(aNet, aeNet)

        json.dump(results, open(os.path.join(dump_dir, file_name), 'w'))


if __name__ == '__main__':
    aeNet = torch.load(f"../pickled/n={nlyr}_en={en}_resblock_3x3conv_5shortcut/de={de}/aeNet_3.pickle", map_location=Meta.device)
    aNet = torch.load(f"../pickled/n={nlyr}_en={en}_resblock_3x3conv_5shortcut/de={de}/aNet.pickle", map_location=Meta.device)

    conditions = attentions_and_probe()
    do_dump(aNet, aeNet, conditions)
