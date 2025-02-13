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

def finite_diff_2nd(func,x,order=4,step=1e-5):
    if order==4:
        diff_func = np.real(-1/12*func(x-2*step)+4/3*func(x-step)-5/2*func(x)+4/3*func(x+step)-1/12*func(x+2*step))/step**2 + 1j*np.imag(-1/12*func(x-2*step)+4/3*func(x-step)-5/2*func(x)+4/3*func(x+step)-1/12*func(x+2*step))/step**2   
    elif order==2:
        diff_func = np.real(func(x-step)-2*func(x)+func(x+step))/step**2 + 1j*np.imag(func(x-step)-2*func(x)+func(x+step))/step**2
    return diff_func

def finite_diff_dxdy(func,x,y,order=4,step=1e-5):
    funcy1 = lambda a : func(x-2*step,a)
    funcy2 = lambda a : func(x-step,a)
    funcy3 = lambda a : func(x+step,a)
    funcy4 = lambda a : func(x+2*step,a)
    diffy1 = finite_diff(funcy1,y)
    diffy2 = finite_diff(funcy2,y)
    diffy3 = finite_diff(funcy3,y)
    diffy4 = finite_diff(funcy4,y)
    if order==4:
        diff = np.real(1/12*diffy1-2/3*diffy2+2/3*diffy3-1/12*diffy4)/step + 1j*np.imag(1/12*diffy1-2/3*diffy2+2/3*diffy3-1/12*diffy4)/step
    elif order==2:
        diff = np.real(-1/2*diffy2+1/2*diffy3)/step + 1j*np.imag(-1/2*diffy2+1/2*diffy3)/step
    return diff

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

def group_vel(omega,alpha,m,Jet,step=1e-5):
    disp_alpha = lambda a: disp(omega,a,m,Jet)
    diff_alpha = finite_diff(disp_alpha,alpha,order=4,step=step)
    disp_omega = lambda o: disp(o,alpha,m,Jet)
    diff_omega = finite_diff(disp_omega,omega,order=4,step=step)
    return -diff_alpha/diff_omega

def build_jacobian(omega,alpha,m,Jet,step):
    disp_alpha = lambda a: disp(omega,a,m,Jet)
    disp_omega = lambda o: disp(o,alpha,m,Jet)
    disp_ao = lambda a, o: disp(a,o,m,Jet)
    
    dalpha = finite_diff(disp_alpha,alpha,order=4,step=step)
    domega = finite_diff(disp_omega,omega,order=4,step=step)

    d2alpha = finite_diff_2nd(disp_alpha,alpha,order=4,step=step)
    d2omega = finite_diff_2nd(disp_omega,omega,order=4,step=step)
    dalphaomega = finite_diff_dxdy(disp_ao,alpha,omega,order=4,step=step)
    
    J = np.zeros((2,2),dtype='complex')
    J[0,0] = dalpha
    J[0,1] = domega
    J[1,0] = (d2alpha*domega-dalpha*dalphaomega)/domega**2
    J[1,1] = (dalphaomega*domega-dalpha*d2omega)/domega**2
    return J

def Newton_tracker(omega,alpha,m,Jet_parameters):
    roots = np.array([alpha,omega])
    search_root = True
    tol=1e-5
    step = 1e-5
    iti=0
    err_ev = []
    while search_root:
        gv_err = group_vel(roots[1], roots[0], m, Jet_parameters,step)
        det_err = disp(roots[1], roots[0], m, Jet_parameters)
        err_norm = np.linalg.norm(np.array([det_err,gv_err]))
        err_ev.append(err_norm)
        if ~np.isnan(err_norm):
            if err_norm<0.1:
                step = 1e-5*err_norm
            else:
                step = 1e-5
            J = build_jacobian(omega,alpha,m,Jet_parameters,step=step) 
            try:
                invJ = np.linalg.inv(J)
                roots = roots - invJ@[det_err, gv_err]
                if err_norm<tol: 
                    search_root = False
                    alpha= roots[0]
                    omega= roots[1]
                elif iti>100: # No convergence
                    # print(f'Step size ={step}, err = {err_norm}')
                    search_root = False
                    if err_norm>1e-2: 
                        # print(f'Lost saddle, err = {err_norm}')
                        omega = np.NaN + 1j*np.nan # Saddle lost
                        alpha = np.NaN + 1j*np.nan
                    else:
                        print(f'Converged saddle, err = {err_norm}')
                        alpha= roots[0]
                        omega = roots[1]
            except Exception as e: 
                search_root = False
                omega = np.NaN + 1j*np.nan # Saddle lost
                alpha = np.NaN + 1j*np.nan
            iti +=1
        else:
            search_root = False
            omega = np.NaN + 1j*np.nan # Saddle lost
            alpha = np.NaN + 1j*np.nan
    return omega, alpha, err_ev

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
    Si = 0.
    Sl = 0.
    Ar = 0.3
    Rout = 0.5
    Ui = -0.3
    Uo = 0
    Ul = 1
    Rin = (1-Ar)*Rout

def main():
    
    m = 1
    Jet = Jet_class()
    tol = 1e-14
    mu = 0.5
    Nguess = 150
    alphas = np.random.normal(0,mu,Nguess) - 1j*np.abs(np.random.normal(0,5*mu,Nguess))
    omegas = np.random.normal(0,mu,Nguess) + 1j*np.abs(np.random.normal(0,2*mu,Nguess))
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

        # omegas_root[i], alphas_root[i], err = Newton_tracker(omegas[i],alphas[i],m,Jet)
        # ax.plot(err)
    # ax.set_xlabel('Iteration')
    # ax.set_ylabel('Error norm')
    # ax.set_yscale('log')
    # ax.set_ylim((1e-3,1e3))
    omegas_filter, alphas_filter = filter_array(omegas_root,alphas_root)
    print(f'Unique saddles found : {len(omegas_filter)}, guesses : {Nguess} ')
    
    # fig, ax = plt.subplots(ncols=2,nrows=1,figsize=(10,6))
    # ax[0].scatter(np.real(alphas_filter),np.imag(alphas_filter),marker='o')
    # ax[1].scatter(np.real(omegas_filter),np.imag(omegas_filter),marker='o')
    # for i in range(len(omegas_filter)):
    #     ax[0].text(np.real(alphas_filter[i]),np.imag(alphas_filter[i]),s=str(i))
    #     ax[1].text(np.real(omegas_filter[i]),np.imag(omegas_filter[i]),s=str(i))
    # ax[0].set_ylabel(r'$\alpha_i$')
    # ax[0].set_xlabel(r'$\alpha_r$')
    # ax[1].set_ylabel(r'$\omega_i$')
    # ax[1].set_xlabel(r'$\omega_r$')
    # ax[1].grid(True)
    # ax[0].grid(True)
    # ax[0].set_xlim(-5,5)
    # ax[0].set_ylim(-7,1)
    # ax[1].set_xlim(-5,5)
    # ax[1].set_ylim(-2,4)
    # fig.tight_layout()
    

    with open('./alpha_Ar03_Uin-03_m1.txt','wb') as f:
        pickle.dump(alphas_filter,f)

    with open('./omega_Ar03_Uin-03_m1.txt','wb') as f:
        pickle.dump(omegas_filter,f)
    
    plt.show()

if __name__=='__main__':
    main()
# def build_grid(limRs, limRi, limIs, limIi, nr, ni):
#     ar = np.linspace(limRi,limRs,nr)
#     ai = np.linspace(limIi,limIs,ni)*1j
#     arr,aii = np.meshgrid(ar,ai)
#     return arr+aii



# def mode_search(omega,m,Jet_parameters, ar_max=6,ai_min=-7,Nr=30,Ni=30):
#     ar_max = 2*omega
#     alpha_grid = build_grid(ar_max,0.01,0,ai_min,Nr,Ni)
#     roots_raw = fsolve_opt(alpha_grid.flatten(),omega,m,Jet_parameters) 
#     num_idx = ~np.isnan(roots_raw)
#     roots_round = np.round(roots_raw[num_idx],decimals=5)
#     roots_filtered = np.unique(roots_round, return_index=False)
#     return roots_filtered

# def saddle_search(wguess,m,Jet,tol=1e-7):
#     alpha0 = []
#     omega0 = []
#     for i in range(len(wguess)):
#         aguess = mode_search(wguess[i],m,Jet)
#         for j in range(len(aguess)):
#             wtrack = wguess[i]
#             atrack = aguess[j]
#             search_root = True; iti=0
#             while search_root:
#                 disperr, _ = dispersion_relation(wtrack, atrack, m, Jet)
#                 gverr = group_velocity_calc(wtrack, atrack, m, Jet)
#                 err_norm = np.linalg.norm(np.array([disperr,gverr]))
#                 J = build_jacobian(wtrack, atrack, m, Jet)
#                 try:
#                     invJ = np.linalg.inv(J)
#                     roots = [atrack, wtrack] - invJ@[disperr, gverr]

#                     alpha_track = roots[0]
#                     omega_track = roots[1]

#                     if err_norm<tol: # Well converged
#                         print(f'Passed low tolerance, err = {err_norm}')
#                         search_root = False
#                         alpha0.append(alpha_track)
#                         omega0.append(omega_track)

#                     elif iti>10: # Not converging
#                         search_root = False
#                         if err_norm<1e-5: # Discart saddle
#                             print(f'Passed high tolerance, err = {err_norm}')
#                             alpha0.append(alpha_track)
#                             omega0.append(omega_track)
#                 #else: # Throw out (I dont have a better solution for now)
#                 except Exception as e: 
#                     print(e)
#                     search_root = False
#                 iti +=1
#     return alpha0, omega0

# omega_grid = build_grid(2,0.01,2,-1,nr=5,ni=5)
# m = 1
# Jet = Jet_class()

# alpha0, omega0 = saddle_search(omega_grid.flatten(),m,Jet)
# print(alpha0)
# print(omega0)

# with open('./alpha_Ar03_Uin-03_m1.txt','wb') as f:
#     pickle.dump(alpha0,f)

# with open('./omega_Ar03_Uin-03_m1.txt','wb') as f:
#     pickle.dump(omega0,f)

# fig, ax = plt.subplots()
# ax.scatter(np.real(alpha0),np.imag(alpha0))
# for x,y,w in zip(np.real(alpha0),np.imag(alpha0),omega0):
#     ax.text(x,y,str(np.round(w,2)))
# ax.grid(True)
# plt.show()

