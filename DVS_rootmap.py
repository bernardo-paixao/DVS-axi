import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time
from functions import dispersion_relation

def contour_roots(omega,m,Jet):
    Nr = 300
    Ni = 500
    alpha_r = np.tile(np.linspace(-1,1,Nr),(Ni,1))
    i_row = np.linspace(-8,1,Ni)*1j
    alpha_i = np.tile(i_row[...,None],(1,Nr))
    alpha_c = alpha_r + alpha_i

    det_min = np.zeros((Ni,Nr))
    for i in range(Ni):
        for j in range(Nr):
            err, _ = dispersion_relation(omega,alpha_c[i,j],m,Jet)
            det_min[i,j] = np.min(np.absolute(err))
    return det_min, alpha_c

class Jet_class:
    Si = 0.
    Sl = 0.
    Ar = 0.3
    Rout = 1
    Ui = -0.3
    Uo = 0
    Ul = 1
    Rin = 1-Ar*Rout


start_time = time.time()
m = 1
omega = 0.5
Jet = Jet_class()

det_err, alpha_contour = contour_roots(omega,m,Jet)
print("--- %s seconds ---" % (time.time() - start_time))

# threshold = 5e-3
# binarized = 1.0 * (det_err > threshold)


fig, ax =plt.subplots()
# ax.imshow(binarized)
pcm = ax.pcolor( np.real(alpha_contour),  np.imag(alpha_contour), det_err,
                   norm=colors.LogNorm(vmin=1e-4, vmax=1e3),
                   cmap='plasma')
fig.colorbar(pcm, ax=ax, extend='max')
ax.set_xlabel('Re(alpha)')
ax.set_ylabel('Im(alpha)')
ax.set_title(f'Map of roots for omega = {omega}, m={m}')

plt.show()




