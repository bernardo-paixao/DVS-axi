import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kv,iv

def Im(m,x):
    return iv(m,x) 

def Km(m,x):
    return kv(m,x)


x = np.linspace(-5,5,200) - 0.2j
order = range(4)

fig, ax = plt.subplots()
for m in order:
    ax.plot(x,np.absolute(Im(m,x)),label=f'm={m}')
ax.set_xlabel('x')
ax.set_ylabel('Im(m,x)')
ax.set_ylim((0,20))
ax.legend()
ax.grid(True)

fig, ax = plt.subplots()
for m in order:
    ax.plot(x,np.absolute(Km(m,x)),label=f'm={m}')
ax.set_xlabel('x')
ax.set_ylabel('Km(m,x)')
ax.set_ylim((0,20))
ax.legend()
ax.grid(True)

print(Km(0,-2+1j))
plt.show()