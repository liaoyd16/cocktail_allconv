###
# Author: Kai Li
# Date: 2021-06-24 19:01:48
# LastEditors: Kai Li
# LastEditTime: 2021-06-26 17:40:22
# Description: file content
###

import os

PROJ_ROOT = os.path.dirname(__file__)
DATA_ROOT = os.path.join(PROJ_ROOT, "dataset/")

def audio_name_(speaker, iblock):
    return os.path.join(DATA_ROOT, "{}/{}.json".format(speaker, iblock))

def mp3_names_(speaker):
    speaker_dir = os.path.join(DATA_ROOT, speaker)
    mp3_names = [name for name in os.listdir(speaker_dir) if name[-4:]=='.mp3']
    return speaker_dir, mp3_names


dump_meta = {
    'dump_layer': 'y5',
}

data_meta = {
    'speakers': ['fem1','fem2','fem3','male1','male2'],
    'pairing': [[1,3], [0,4]], # each is a pair
    
    'blocks_per_speaker': 22,
    'train_blocks_per_speaker': [i for i in range(1, 20+1)],
    'test_blocks_per_speaker': [i for i in range(21, 22+1)],
    'iblock_embedder': 0,

    'secs_per_block': 4*60,
    'secs_per_slice': 4, # 4s
    'slices_per_block': 60, # TODO: secs_per_block // secs_per_block
    'Fs': 22050,
    'slice_len': 22050*4, # TODO: Fs * secs_per_slice
    'specgram_size': (256, 128),
    'fft_size': 2048,
    'step_size': 128,
    'spec_thresh': 0,
    'freq_range': [200, 3000]
}

assert(data_meta['secs_per_block'] == data_meta['secs_per_slice'] * data_meta['slices_per_block'])
assert(data_meta['slice_len'] == data_meta['Fs'] * data_meta['secs_per_slice'])


import numpy as np
def get_using_speakers():
    using_speakers = [False for speaker in data_meta['speakers']]
    for (sp1, sp2) in data_meta['pairing']:
        using_speakers[sp1] = True
        using_speakers[sp2] = True

    return np.array(data_meta['speakers'])[using_speakers]


model_meta = {
    'feature_vector_size': 256,
    'feature_net_classes': 5,
    'attend_layers': 5, ## TODO: 3 / 5 / 8
    'layer_attentions': [1,1,1,1,1], ## TODO: full/bot-up/top-dwn
    'keepsize_num_en': 1,
    'keepsize_num_de': 0,
    'embedder': {
        'sr': 16000,
        'window': 0.025,
        'hop': 0.01,
        'nmels': 40,
        'hidden': 768,
        'num_layer': 3,
        'proj': 256,
        'tisv_frame': 180,
        'nfft': 512,
    }
}

assert(len(model_meta['layer_attentions']) == model_meta['attend_layers'])

import torch

EPS = 1e-5

def lg(x):
    x[x <= 0] = EPS
    return np.log(x) / np.log(10)

def mel(specgram):
    specgram[specgram < 0] = 0
    return lg(1 + specgram/ 4)

DEVICE_ID = "cuda:0"
device = torch.device(DEVICE_ID if torch.cuda.is_available() else "cpu")
