"""
import matplotlib.pyplot as plt
import numpy as np

# some data
x = np.arange(0, 10, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

# plot of the data
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
ax.plot(x, y1,'-k', lw=2, label='black sin(x)')
ax.plot(x, y2,'-r', lw=2, label='red cos(x)')
ax.set_xlabel('x', size=22)
ax.set_ylabel('y', size=22)
ax.legend(bbox_to_anchor=(1.1, .5), loc='center left', borderaxespad=0.)

plt.show()
"""
import __init__
from __init__ import *
from aux import _load_features_of_speakers

ans = _load_features_of_speakers()
fem2 = ans['fem2']
male1 = ans['male1']
fem1 = ans['fem1']
male2 = ans['male2']

diff = fem2 - fem1
mean = (fem2 + fem1) / 2
print(torch.norm(diff) / torch.norm(mean))

diff = male1 - male2
mean = (male1 + male2) / 2
print(torch.norm(diff) / torch.norm(mean))
