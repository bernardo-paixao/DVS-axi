import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kv,iv
from functions import dispersion_relation, null

def get_eigenvector(alpha_target, omega_target, m, Jet, Nr = 250):
    err, M = dispersion_relation(omega_target,alpha_target,m,Jet)
    if err>1e-10:
        print(f'Warning: M determinant = {np.round(err,5)}') 

    Coeff_vect = null(M)
    Do = Coeff_vect[0]
    Cl = Coeff_vect[1]
    Dl = Coeff_vect[2]
    Ci = Coeff_vect[3]

    r = np.linspace(0,2,Nr)
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
    # testdet = M[0,0]*(M[1,1]*M[2,2]*M[3,3]+M[1,2]*M[2,3]*M[3,1]-M[1,2]*M[2,1]*M[3,3]-M[2,3]*M[3,2]*M[1,1])-M[1,0]*(M[0,1]*M[2,2]*M[3,3]+M[0,2]*M[2,3]*M[3,1]-M[0,2]*M[2,1]*M[3,3]-M[2,3]*M[3,2]*M[0,1])
    return det

class Jet_class:
    Si = 0.
    Sl = 0.1
    Ar = 0.1
    Rout = 0.5
    Ui = -0.3
    Uo = 0
    Ul = 1
    Rin = (1-Ar)*Rout

def main():
    
    m = 1
    St = 0.05
    omega = (2*np.pi)*St
    Jet = Jet_class()

    disp_alpha = lambda a: disp(omega,a,m,Jet)

    tol = 1e-14
    # fig, ax = plt.subplots()
    colors = ['k','r','b']
    count = 0
    ar = np.linspace(0,10*omega,40)
    ai = -np.linspace(0,10*omega,40)
    arr, aii = np.meshgrid(ar,ai)
    alphas = (arr+1j*aii).flatten()

    alphas_root = np.zeros_like(alphas)
    for j, alpha_guess in enumerate(alphas):
        iti = 0
        search_root = True
        err = 1
        err_list = []
        while search_root:
            if ~np.isnan(alpha_guess):
                diff_disp = finite_diff(disp_alpha,alpha_guess,order=4,step=1e-5)
                alpha_guess = alpha_guess - disp_alpha(alpha_guess)/diff_disp
                err =  np.absolute(disp_alpha(alpha_guess))
                err_list.append(err)
                if np.abs(np.real(alpha_guess))>50*omega:
                    alphas_root[j] = np.NaN + 1j*np.NaN
                    search_root = False
                if err<tol:
                    search_root = False
                    alphas_root[j] = alpha_guess
                    # print(f'Success, alpha={alpha_guess}, err = {err}')
                    count +=1
                if iti>15: 
                    alphas_root[j] = np.NaN + 1j*np.NaN
                    search_root = False
                iti +=1
            else:
                search_root = False
                alphas_root[j] = np.NaN + 1j*np.NaN
        
        # ax.plot(np.arange(iti)+1,err_list,ls='',marker='.',c=colors[i],label=f'order={(i+1)*2}')
        # ax.set_yscale('log')

    print(f'Success rate order 4 : {count/10}%')
    fig, ax = plt.subplots()
    ax.scatter(np.real(alphas),np.imag(alphas),marker='.',c='b')
    ax.scatter(np.real(alphas_root),np.imag(alphas_root),c='r')
    ax.grid()
    # plt.show()

    roots_filtered = np.unique(alphas_root.round(5), return_index=False)
    roots_filtered = roots_filtered[~np.isnan(roots_filtered)]
    # alpha = []
    # for m in m_list: 
    #     start_time = time.time()
    #     roots = mode_search(omega,m,Jet)
    #     print("--- %s seconds ---" % (time.time() - start_time))
    #     alpha.append(roots)

    # fig,ax = plt.subplots()
    # for i in range(len(m_list)):
    #     ax.scatter(np.real(alpha[i]),np.imag(alpha[i]),marker='o',label='m='+str(m_list[i]))
    # ax.set_ylabel(r'$\alpha_i$')
    # ax.set_xlabel(r'$\alpha_r$')
    # ax.legend()
    # ax.grid(True)
    Ntheta = 100
    theta = np.linspace(0,2*np.pi,Ntheta)
    # for j in range(len(m_list)):
    for i in range(len(roots_filtered)):
        p, r = get_eigenvector(roots_filtered[i],omega,m,Jet)
        if i ==0:
            RR, Theta = np.meshgrid(r,theta)
            X =RR*np.cos(Theta) 
            Y =RR*np.sin(Theta) 
        
        P = np.tile(p,(Ntheta,1))*np.exp(1j*m*Theta)
        fig, ax = plt.subplots()
        vlim = np.max(np.abs(P))
        cb = ax.pcolor(X,Y,np.real(P), cmap='bwr',vmin=-vlim, vmax=vlim)
        plt.colorbar(cb,label=r'$\phi_z$')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        # ax.plot(r,np.absolute(p)/np.max(np.absolute(p)),color='k')
        # ax.plot(r,np.real(p)/np.max(np.absolute(p)),color='b')
        # ax.plot(r,np.imag(p)/np.max(np.absolute(p)),color='r')
        # ax.plot(Jet.Rin*np.ones(100),np.linspace(-1.2,1.2,100),ls='--',color='grey')
        # ax.plot(Jet.Rout*np.ones(100),np.linspace(-1.2,1.2,100),ls='--',color='grey')
        # ax.set_xlabel('r')
        # ax.set_ylabel('p')
        # ax.set_ylim(-1.1,1.1)
        ax.set_xlim(-0.7,0.7)
        ax.set_ylim(-0.7,0.7)
        # ax.grid(True)
        ax.set_title(f'alpha = {np.round(roots_filtered[i],2)}, omega={omega}, m={m}')
    plt.show()

if __name__=='__main__':
    main()