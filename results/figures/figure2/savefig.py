import numpy as np
import json
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import interp2d

FPAND = 3
TPAND = 10

sp1 = np.array(json.load(open("pair_0/sp1.json")))[::-1]
sp2 = np.array(json.load(open("pair_0/sp2.json")))[::-1]
mix = np.array(json.load(open("pair_0/mix.json")))[::-1]
none = np.array(json.load(open("pair_0/re_none.json")))[::-1]
re1 = np.array(json.load(open("pair_0/re1.json")))[::-1]
re2 = np.array(json.load(open("pair_0/re2.json")))[::-1]

def interpolate(img, xpand, ypand):
    size0 = img.shape
    size1 = (img.shape[0]*xpand, img.shape[1]*ypand)

    temp_ind = np.arange(size1[1])
    freq_ind = np.arange(size1[0])
    ffttemp2ind = np.linspace(0, size1[1], num=size0[1])
    fftfreq2ind = np.linspace(0, size1[0], num=size0[0])

    return interp2d(y=fftfreq2ind, x=ffttemp2ind, z=img)(temp_ind, freq_ind)


sp1 = interpolate(sp1, FPAND, TPAND)
sp2 = interpolate(sp2, FPAND, TPAND)
mix = interpolate(mix, FPAND, TPAND)
none = interpolate(none, FPAND, TPAND)
re1 = interpolate(re1, FPAND, TPAND)
re2 = interpolate(re2, FPAND, TPAND)

def do_ticks():
    plt.xticks([0, 31*TPAND, 63*TPAND, 95*TPAND, 127*TPAND], ['0s','1s','2s','3s','4s'])
    plt.yticks([0, 63*FPAND, 127*FPAND, 191*FPAND, 255*FPAND], ['200Hz', '900Hz', '1600Hz', '2300Hz', '3000Hz'][::-1])


plt.imshow(np.array(sp1), cmap=plt.cm.jet)
do_ticks()
plt.savefig("sp1.png", dpi=300)
plt.close()

plt.imshow(np.array(sp2), cmap=plt.cm.jet)
do_ticks()
plt.savefig("sp2.png", dpi=300)
plt.close()

plt.imshow(np.array(re1), cmap=plt.cm.jet)
do_ticks()
plt.savefig("re1.png", dpi=300)
plt.close()

plt.imshow(np.array(re2), cmap=plt.cm.jet)
do_ticks()
plt.savefig("re2.png", dpi=300)
plt.close()

plt.imshow(np.array(none), cmap=plt.cm.jet)
do_ticks()
plt.savefig("none.png", dpi=300)
plt.close()

plt.imshow(np.array(mix), cmap=plt.cm.jet)
do_ticks()
plt.savefig("mix.png", dpi=300)
plt.close()

sp12 = np.array([sp1, sp2, np.zeros((256*FPAND,128*TPAND), dtype=int)])
do_ticks()
plt.imshow(1 - sp12.transpose(1,2,0))
plt.savefig("mix-color.png", dpi=300)
plt.close()

