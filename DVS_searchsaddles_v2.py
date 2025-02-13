import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kv,iv
from scipy.optimize import fsolve
from functions import dispersion_relation, null
import pickle

def get_eigenvector(alpha_target, omega_target, m, Jet, Nr = 250):
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

def finite_diff(func,x,order=4,step=1e-5):
    if order==6:
        diff_func = np.real(-1/60*func(x-3*step)+3/20*func(x-2*step)-3/4*func(x-step)+3/4*func(x+step)-3/20*func(x+2*step)+1/60*func(x-3*step))/step + 1j*np.imag(-1/60*func(x-3*step)+3/20*func(x-2*step)-3/4*func(x-step)+3/4*func(x+step)-3/20*func(x+2*step)+1/60*func(x-3*step))/step
    elif order==4:
        diff_func = np.real(1/12*func(x-2*step)-2/3*func(x-step)+2/3*func(x+step)-1/12*func(x+2*step))/step + 1j*np.imag(1/12*func(x-2*step)-2/3*func(x-step)+2/3*func(x+step)-1/12*func(x+2*step))/step   
    elif order==2:
        diff_func = np.real(-1/2*func(x-step)+1/2*func(x+step))/step + 1j*np.imag(-1/2*func(x-step)+1/2*func(x+step))/step
    return diff_func

def Im(m,x):
    return iv(m,x) 

def Km(m,x):
    return kv(m,x)

def beta(omega,alpha,m,Jet,type):
    if type=='i':   
        D = 1j*(-omega+m*Jet.Si+alpha*Jet.Ui)
        beta = alpha*np.sqrt(D**2+4*Jet.Si**2)/D
    elif type=='l':    
        D = 1j*(-omega+m*Jet.Sl+alpha*Jet.Ul)
        beta = alpha*np.sqrt(D**2+4*Jet.Sl**2)/D
    else:
        beta= np.sign(np.real(alpha))*alpha
    return beta   

def disp(omega,alpha,m,Jet):
    M = np.zeros((4,4),dtype='complex')

    Do = 1j*(-omega+alpha*Jet.Uo)
    Dl = 1j*(-omega+m*Jet.Sl+alpha*Jet.Ul)
    Di = 1j*(-omega+m*Jet.Si+alpha*Jet.Ui)
    beta_o = np.sign(np.real(alpha))*alpha
    beta_l = alpha*np.sqrt(Dl**2+4*Jet.Sl**2)/Dl
    beta_i = alpha*np.sqrt(Di**2+4*Jet.Si**2)/Di

    M[0,0] = Km(m,beta_o*Jet.Rout)
    M[0,1] = -Im(m,beta_l*Jet.Rout)
    M[0,2] = -Km(m,beta_l*Jet.Rout)
    M[1,0] = beta_o*(Km(m-1,beta_o*Jet.Rout)+Km(m+1,beta_o*Jet.Rout))/(2*Do**2)
    M[1,1] = beta_l*(Im(m-1,beta_l*Jet.Rout)+Im(m+1,beta_l*Jet.Rout))/(2*(Dl**2+4*Jet.Sl**2))+1j*m*2*Jet.Sl*Im(m,beta_l*Jet.Rout)/(Jet.Rout*Dl*(Dl**2+4*Jet.Sl**2))
    M[1,2] = -beta_l*(Km(m-1,beta_l*Jet.Rout)+Km(m+1,beta_l*Jet.Rout))/(2*(Dl**2+4*Jet.Sl**2))+1j*m*2*Jet.Sl*Km(m,beta_l*Jet.Rout)/(Jet.Rout*Dl*(Dl**2+4*Jet.Sl**2))
    M[2,1] = Im(m,beta_l*Jet.Rin)
    M[2,2] = Km(m,beta_l*Jet.Rin)
    M[2,3] = -Im(m,beta_i*Jet.Rin)
    M[3,1] = -beta_l*(Im(m-1,beta_l*Jet.Rin)+Im(m+1,beta_l*Jet.Rin))/(2*(Dl**2+4*Jet.Sl**2))-1j*m*2*Jet.Sl*Im(m,beta_l*Jet.Rin)/(Jet.Rin*Dl*(Dl**2+4*Jet.Sl**2))
    M[3,2] = beta_l*(Km(m-1,beta_l*Jet.Rin)+Km(m+1,beta_l*Jet.Rin))/(2*(Dl**2+4*Jet.Sl**2))-1j*m*2*Jet.Sl*Km(m,beta_l*Jet.Rin)/(Jet.Rin*Dl*(Dl**2+4*Jet.Sl**2))
    M[3,3] = beta_i*(Im(m-1,beta_i*Jet.Rin)+Im(m+1,beta_i*Jet.Rin))/(2*(Di**2+4*Jet.Si**2))+1j*m*2*Jet.Si*Im(m,beta_i*Jet.Rin)/(Jet.Rin*Di*(Di**2+4*Jet.Si**2))
    det = np.linalg.det(M)
    return det

def group_vel(omega,alpha,m,Jet,step=1e-5):
    disp_alpha = lambda a: disp(omega,a,m,Jet)
    diff_alpha = finite_diff(disp_alpha,alpha,order=4,step=step)
    disp_omega = lambda o: disp(o,alpha,m,Jet)
    diff_omega = finite_diff(disp_omega,omega,order=4,step=step)
    return -diff_alpha/diff_omega

def fsaddle(vec,m,Jet):
    omega = vec[0]+vec[1]*1j
    alpha = vec[2]+vec[3]*1j
    d = disp(omega,alpha,m,Jet)
    g = group_vel(omega,alpha,m,Jet) 
    return [np.real(d),np.imag(d),np.real(g),np.imag(g)]

def filter_array(a,b):
    num_idx = ~np.isnan(a)
    num_idx2 = ~np.isnan(b)
    if (num_idx==num_idx2).all():
        print('Ok')
    else:
        print('Alpha and omega NaN are different')
    a = a[num_idx]
    b = b[num_idx]
    round_a = np.round(a,decimals=5)
    _, unique_idx = np.unique(round_a, return_index=True)
   
    return a[unique_idx], b[unique_idx]

class Jet_class:
    Si = 0.25
    Sl = 0.25
    Ar = 0.1
    Rout = 0.5
    Ui = -0.3
    Uo = 0
    Ul = 1
    Rin = (1-Ar)*Rout

def main():
    m = 1
    Jet = Jet_class()

    Nguess = 200
    a_guess = 3-2j
    o_guess = 3+0j
    radius = 3

    alphas = a_guess + radius*(2*(np.random.rand(Nguess)-0.5) + 2j*(np.random.rand(Nguess)-0.5))
    omegas = o_guess + radius*(2*(np.random.rand(Nguess)-0.5) + 2j*(np.random.rand(Nguess)-0.5))

    fig, ax = plt.subplots(ncols=2)
    ax[0].plot(np.real(alphas),np.imag(alphas),ls='',marker='.')
    ax[1].plot(np.real(omegas),np.imag(omegas),ls='',marker='.')
    for i in range(2):
        ax[i].grid(True)

    ax[0].set_xlabel(r'$Re(\alpha)$')
    ax[0].set_ylabel(r'$Im(\alpha)$')
    ax[1].set_xlabel(r'$Re(\omega)$')
    ax[1].set_ylabel(r'$Im(\omega)$')

    alphas_root = np.zeros_like(alphas)
    omegas_root = np.zeros_like(omegas)

    # fig, ax = plt.subplots()
    for i in range(Nguess):
        roots = fsolve(fsaddle,[np.real(omegas[i]),np.imag(omegas[i]),np.real(alphas[i]),np.imag(alphas[i])],args=(m,Jet),xtol=1e-06)
        err = np.linalg.norm(fsaddle(roots,m,Jet))
        if err<1e-5:
            omegas_root[i]=roots[0]+roots[1]*1j
            alphas_root[i] = roots[2]+roots[3]*1j
            # print(f'alpha = {alphas_root[i].round(3)}, growth rate = {np.imag(omegas_root[i]).round(3)}')
        else:
            omegas_root[i] = np.nan
            alphas_root[i] = np.nan


    omegas_filter, alphas_filter = filter_array(omegas_root,alphas_root)

    o_select = (np.abs(np.imag(omegas_filter))<5)*(np.real(omegas_filter)<8)*(np.real(omegas_filter)>0)*(np.abs(alphas_filter)<10)
    omegas_filter = omegas_filter[o_select]
    alphas_filter = alphas_filter[o_select]
    Nselec = len(omegas_filter)

    print(f'Selected saddles : {Nselec}, guesses : {Nguess} ')

    fig, ax = plt.subplots()
    ax.plot(np.real(alphas_filter),np.imag(alphas_filter),ls='',marker='.')
    for i, txt in enumerate(omegas_filter):
        ax.annotate(str(txt.round(2)), (np.real(alphas_filter[i]), np.imag(alphas_filter[i])))
    ax.set_xlabel(r'$Re(\alpha)$')
    ax.set_ylabel(r'$Im(\alpha)$')
    ax.grid(True)

    Np = 50
    Ui_list = -np.linspace(-0.25,0,Np)
    omega_evo = np.zeros((Nselec,Np),dtype='complex')
    alpha_evo = np.zeros((Nselec,Np),dtype='complex')

    for i in range(Np):
        Jet.Si = Ui_list[i]
        Jet.Sl = Ui_list[i]
        for j in range(Nselec):
            if i==0:
                x0 = [np.real(omegas_filter[j]),np.imag(omegas_filter[j]),np.real(alphas_filter[j]),np.imag(alphas_filter[j])]              
            else:
                x0 = [np.real(omega_evo[j,i-1]),np.imag(omega_evo[j,i-1]),np.real(alpha_evo[j,i-1]),np.imag(alpha_evo[j,i-1])]  

            roots = fsolve(fsaddle,x0,args=(m,Jet),xtol=1e-06)
            omega_evo[j,i]=roots[0]+roots[1]*1j
            alpha_evo[j,i] = roots[2]+roots[3]*1j
            if i==0:
                print(f'Guess : {alphas_filter[j]}')
                print(f'Shift from guess : {alpha_evo[j,i]-alphas_filter[j]}')
      

    
    fig, ax = plt.subplots()
    for j in range(Nselec):
        ax.plot(np.real(alpha_evo[j,:]),np.imag(alpha_evo[j,:]),ls='',marker='.')
    ax.set_xlabel(r'$Re(\alpha)$')
    ax.set_ylabel(r'$Im(\alpha)$')
    ax.grid(True)
    
    with open('./alpha_Ar01_Ui03_S_m1.txt','wb') as f:
        pickle.dump(alpha_evo,f)

    with open('./omega_Ar01_Ui03_S_m1.txt','wb') as f:
        pickle.dump(omega_evo,f)

    
    plt.show()

if __name__=='__main__':
    main()

