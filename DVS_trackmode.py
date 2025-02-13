import numpy as np
import matplotlib.pyplot as plt
import time
from functions import *
from functools import partial
import multiprocessing as mp
import pickle

class Jet_class:
    Si = 0.
    Sl = 0.
    Ar = 0.3
    Rout = 1
    Ui = -0.2
    Uo = 0
    Ul = 1
    Rin = 1-Ar*Rout

def get_eigenvector(alpha_target, omega_target, m, Jet, Nr = 1000):
    err, M = dispersion_relation(omega_target,alpha_target,m,Jet)
    if err>1e-10:
        print(f'Warning: M determinant = {np.round(err,5)}') 

    Coeff_vect = null(M)
    Do = Coeff_vect[0]
    Cl = Coeff_vect[1]
    Dl = Coeff_vect[2]
    Ci = Coeff_vect[3]

    r = np.linspace(0,5,Nr)
    p = np.zeros(Nr,dtype='complex')
    for i in range(Nr):
        if r[i]<Jet.Rin:
            beta_i = beta(omega_target,alpha_target,m,Jet,type='i')
            p[i] = Ci*Im(m,beta_i*r[i])
        elif r[i]>=Jet.Rin and r[i]<=Jet.Rout:
            beta_l = beta(omega_target,alpha_target,m,Jet,type='l')
            p[i] = Cl*Im(m,beta_l*r[i])+Dl*Km(m,beta_l*r[i])
        else:
            beta_o = beta(omega_target,alpha_target,m,Jet,type='o')
            p[i] = Do*Km(m,beta_o*r[i]) 
    return p, r
    

if __name__ == '__main__': 
    start  = time.perf_counter()
    pool = mp.Pool()

    omega = [0.3] #np.linspace(0.001,0.3,3)
    m = [0,1,2]
    Jet = Jet_class()
    alphas = []

    for j in range(len(m)):
        func_handle = partial(mode_search,m=m[j],Jet_parameters=Jet)
        results = pool.map(func_handle, omega)
        alphas.append(results)
        print(f'Finished m={m[j]}')

    file_name = 'stab_analysis_Ar03_Uin02_S0_3.txt'

    with open('./Data/alpha_'+file_name,'wb') as f:
        pickle.dump(alphas,f)

    with open('./Data/info_'+file_name,'w') as f:
        f.write('Ar = '+str(Jet.Ar)+'\n Uin = '+str(Jet.Ui)+' \n S = '+str(Jet.Sl)+' \n m = '+str(m)+' \n omega = '+str(omega))

    # c = ['k','r','b','g','m']
    c = plt.cm.viridis(np.linspace(0,1,len(omega)))
    markers = ['.','*','+','s','d']
    fig, ax = plt.subplots()
    # for j in range(len(m)):
        # for i in range(len(omega)):
        #     # alphas.append(mode_search(omega[i],m[j],Jet))
        #     ax.scatter(np.real(alphas[i]),np.imag(alphas[i]),marker=markers[j],color=c[i])
        #     # if m[i]!=0:
        #     #     alphas.append(mode_search(-omega,m[i],Jet))
        #     #     ax.scatter(-np.real(alphas[i+1]),np.imag(alphas[i+1]),marker='s',color=c[i],label=-m[i])
        #     # alphas.append(fsolve_opt(alpha_guess,omega,m[i],Jet))
        # ax.legend()
    for j in range(len(m)):
        i=0
        for alpha in alphas[j]:
            ax.scatter(np.real(alpha),np.imag(alpha),marker=markers[j],color=c[i])
            i+=1
    
    ax.set_xlim((-2,7))
    ax.set_ylim((-8,0))
    ax.set_ylabel(r'$\alpha_i$')
    ax.set_xlabel(r'$\alpha_r$')
    ax.grid(True)

    finish = time.perf_counter()
    print(f'Finished in {round(finish-start,2)} second(s)')
    # plt.show()


    # for i in range(len(alpha_guess)):
    #     ax.plot(err_list[i,:])
    #     ax.set_yscale('log')

    for j in range(len(m)):
        for i in range(len(alphas[j])):
            #if np.absolute(alphas[j][i])<10:
            p, r = get_eigenvector(alphas[j][i],omega,m[j],Jet)
            fig, ax = plt.subplots()
            ax.plot(r,np.absolute(p)/np.max(np.absolute(p)),color='k')
            ax.plot(r,np.real(p)/np.max(np.absolute(p)),color='b')
            ax.plot(r,np.imag(p)/np.max(np.absolute(p)),color='r')
            ax.plot(Jet.Rin*np.ones(100),np.linspace(-1.2,1.2,100),ls='--',color='grey')
            ax.plot(Jet.Rout*np.ones(100),np.linspace(-1.2,1.2,100),ls='--',color='grey')
            ax.set_xlabel('r')
            ax.set_ylabel('p')
            ax.set_ylim(-1.1,1.1)
            ax.set_xlim(0,5)
            ax.grid(True)
            ax.set_title(f'alpha = {np.round(alphas[j][i],2)}, omega={omega}, m={m[j]}')
            fig.savefig(f'Jet_Ar03_Uin-02_S0_m{m[j]}_{i}_omega=01')
            plt.close(fig)


    # print(alphas)