
import __init__
from __init__ import *

import Meta
fft_size = Meta.data_meta['fft_size']
Fs = Meta.data_meta['Fs']
step_size = Meta.data_meta['step_size']
freq_range = Meta.data_meta['freq_range']
spec_size = Meta.data_meta['specgram_size']

fft_outsize = [260, 690] # TODO: 173 for 1s, 690 for 4s

spec_converter = torchaudio.transforms.Spectrogram(n_fft=fft_size, hop_length=step_size)

freqs = np.linspace(0, Fs//2, num=fft_size//2+1)
bandpass = np.logical_and(freqs >= freq_range[0], freqs <= freq_range[1])

import scipy.interpolate
def do_interpolate(specs):
    temp_ind = np.arange(spec_size[1])
    freq_ind = np.arange(spec_size[0])
    ffttemp2ind = np.linspace(0, spec_size[1], num=fft_outsize[1])
    fftfreq2ind = np.linspace(0, spec_size[0], num=fft_outsize[0])
    ans = np.array([scipy.interpolate.interp2d(y=fftfreq2ind, x=ffttemp2ind, z=spec)(temp_ind, freq_ind) 
                    for spec in specs])
    return ans


def spectrogram(slices): # BS * LEN
    # BS x (NFFT/2+1) x TEMP -> BS x BAND x TEMP -> log scale
    slices = torch.Tensor(slices)
    specs = spec_converter(slices)
    specs = torch.log10_(specs[:, bandpass])
    specs = do_interpolate(specs)
    specs[specs < 0] = 0
    return specs


import matplotlib.pyplot as plt
def display_spec(spec):
    spec[spec < 0] = 0
    im = plt.imshow(spec)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.show()


if __name__ == '__main__':
    import time

    xs, _ = torchaudio.load("/Users/liaoyuanda/Desktop/cocktail-misc/cocktail/dataset/sp0/2.wav")
    xs4 = xs[0,:4*Fs].numpy()

    test_bs = 10
    nbatch = 50
    test_input = [xs4 for _ in range(test_bs)]
    
    start = time.time()
    for i in range(nbatch):
        specs = spectrogram(test_input)
    end = time.time()
    print(end - start)
