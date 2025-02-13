import numpy as np
from scipy.special import kv,iv

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

def dispersion_relation(omega,alpha,m,Jet):
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
    return det, M
    
# def dispersion_relation(omega,alpha,m,Jet): #modified
#     M = np.zeros((4,4),dtype='complex')

#     Do = 1j*(-omega+alpha*Jet.Uo)
#     Dl = 1j*(-omega+m*Jet.Sl+alpha*Jet.Ul)
#     Di = 1j*(-omega+m*Jet.Si+alpha*Jet.Ui)
#     beta_o = np.sign(np.real(alpha))*alpha
#     beta_l = alpha*np.sqrt(Dl**2+4*Jet.Sl**2)/Dl
#     beta_i = alpha*np.sqrt(Di**2+4*Jet.Si**2)/Di

#     M[0,0] = Km(m,beta_o*Jet.Rout)
#     M[0,1] = -Im(m,beta_l*Jet.Rout)
#     M[0,2] = -Km(m,beta_l*Jet.Rout)
#     M[1,0] = beta_o*(Km(m-1,beta_o*Jet.Rout)+Km(m+1,beta_o*Jet.Rout))/(2*Do**2)*Dl*(Dl**2+4*Jet.Sl**2)
#     M[1,1] = beta_l*(Im(m-1,beta_l*Jet.Rout)+Im(m+1,beta_l*Jet.Rout))*Dl/2+1j*m*2*Jet.Sl*Im(m,beta_l*Jet.Rout)/Jet.Rout
#     M[1,2] = -beta_l*(Km(m-1,beta_l*Jet.Rout)+Km(m+1,beta_l*Jet.Rout))*Dl/2+1j*m*2*Jet.Sl*Km(m,beta_l*Jet.Rout)/Jet.Rout
#     M[2,1] = Im(m,beta_l*Jet.Rin)
#     M[2,2] = Km(m,beta_l*Jet.Rin)
#     M[2,3] = -Im(m,beta_i*Jet.Rin)
#     M[3,1] = -beta_l*(Im(m-1,beta_l*Jet.Rin)+Im(m+1,beta_l*Jet.Rin))*Dl*Di*(Di**2+4*Jet.Si**2)/2-1j*m*2*Jet.Sl*Im(m,beta_l*Jet.Rin)*Di*(Di**2+4*Jet.Si**2)/Jet.Rin
#     M[3,2] = beta_l*(Km(m-1,beta_l*Jet.Rin)+Km(m+1,beta_l*Jet.Rin))*Dl*Di*(Di**2+4*Jet.Si**2)/2-1j*m*2*Jet.Sl*Km(m,beta_l*Jet.Rin)*Di*(Di**2+4*Jet.Si**2)/Jet.Rin
#     M[3,3] = beta_i*(Im(m-1,beta_i*Jet.Rin)+Im(m+1,beta_i*Jet.Rin))*Dl*(Dl**2+4*Jet.Sl**2)*Di/2+1j*m*2*Jet.Si*Im(m,beta_i*Jet.Rin)*Dl*(Dl**2+4*Jet.Sl**2)/Jet.Rin
#     det = np.linalg.det(M)
#     # testdet = M[0,0]*(M[1,1]*M[2,2]*M[3,3]+M[1,2]*M[2,3]*M[3,1]-M[1,2]*M[2,1]*M[3,3]-M[2,3]*M[3,2]*M[1,1])-M[1,0]*(M[0,1]*M[2,2]*M[3,3]+M[0,2]*M[2,3]*M[3,1]-M[0,2]*M[2,1]*M[3,3]-M[2,3]*M[3,2]*M[0,1])
#     return det,M

def complex_diff(omega,alpha,m,Jet_parameters,diffomega=False):
    N=5
    hr = 1e-5
    det_err = np.zeros(N,dtype='complex')
    near_alpha = np.linspace(np.real(alpha)-2*hr,np.real(alpha)+2*hr,N)+1j*np.imag(alpha)

    for i in range(N):
        det_err[i], _ = dispersion_relation(omega,near_alpha[i],m,Jet_parameters)
    diff_alpha = np.real(1/12*det_err[0]-2/3*det_err[1]+2/3*det_err[3]-1/12*det_err[4])/hr + 1j*np.imag(1/12*det_err[0]-2/3*det_err[1]+2/3*det_err[3]-1/12*det_err[4])/hr
    
    if diffomega==True:
        near_omega = np.linspace(np.real(omega)-2*hr,np.real(omega)+2*hr,N)+1j*np.imag(omega)
        for i in range(N):
            det_err[i], _ = dispersion_relation(near_omega[i],alpha, m,Jet_parameters)
        diff_omega = np.real(1/12*det_err[0]-2/3*det_err[1]+2/3*det_err[3]-1/12*det_err[4])/hr + 1j*np.imag(1/12*det_err[0]-2/3*det_err[1]+2/3*det_err[3]-1/12*det_err[4])/hr
        return diff_alpha, diff_omega
    else:
        return diff_alpha

def fsolve_opt(guess_list, omega,m,Jet_parameters):
    tol = 1e-14
    Nguess = len(guess_list)
    root = np.copy(guess_list)
    err_list = np.zeros((Nguess,18))
    for i in range(Nguess):
        iti = 0
        search_root = True
        err = 1
        while search_root:
            if ~np.isnan(root[i]):
                diff_det = complex_diff(omega,root[i],m,Jet_parameters)
                det, _ = dispersion_relation(omega,root[i],m,Jet_parameters)
                root[i] = root[i] - det/diff_det
                err =  np.absolute(det)
                # err_list[i,iti] = err
                if err<tol:
                    search_root = False
                if iti>15: # It was 25
                    # if err>1e-5:
                        # print(f'Warning, maximum itiration number achieved, err = {err}')
                    root[i] = np.NaN + 1j*np.NaN
                    search_root = False
                iti +=1
            else:
                search_root = False
    return root

def null(A):
    w , v = np.linalg.eig(A)
    idx_min = np.argmin(np.absolute(w))
    null_space = v[:,idx_min]
    err = np.linalg.norm(A@null_space)
    print(f'Norm M x Null space = {err}')
    return null_space.T

def group_velocity_calc(omega,alpha,m,Jet_parameters):
    diff_alpha, diff_omega = complex_diff(omega,alpha,m,Jet_parameters,diffomega=True)
    return -diff_alpha/diff_omega

def dd_alpha(omega,alpha,m,Jet_parameters):
    N=5
    hr = 1e-6
    det_err = np.zeros(N,dtype='complex')
    near_alpha_r = np.linspace(np.real(alpha)-2*hr,np.real(alpha)+2*hr,N)+1j*np.imag(alpha)
    for i in range(N):
        det_err[i], _ = dispersion_relation(omega, near_alpha_r[i], m, Jet_parameters)
    diff = np.real(-1/12*det_err[0]+4/3*det_err[1]-5/2*det_err[2]+4/3*det_err[3]-1/12*det_err[4])/hr**2 + 1j*np.imag(-1/12*det_err[0]+4/3*det_err[1]-5/2*det_err[2]+4/3*det_err[3]-1/12*det_err[4])/hr**2
    # diff = np.real(det_err[2]-2*det_err[1]+det_err[0])/(hr**2) + 1j*np.imag(det_err[2]-2*det_err[1]+det_err[0])/(hr**2) # Accuracy 2, set N=3
    return diff

def dd_omega(omega,alpha,m,Jet_parameters):
    N=5
    hr = 1e-6
    det_err = np.zeros(N,dtype='complex')
    near_omega_r = np.linspace(np.real(omega)-2*hr,np.real(omega)+2*hr,N)+1j*np.imag(omega)
    for i in range(N):
        det_err[i], _ = dispersion_relation(near_omega_r[i], alpha, m, Jet_parameters)
    diff = np.real(-1/12*det_err[0]+4/3*det_err[1]-5/2*det_err[2]+4/3*det_err[3]-1/12*det_err[4])/hr**2 + 1j*np.imag(-1/12*det_err[0]+4/3*det_err[1]-5/2*det_err[2]+4/3*det_err[3]-1/12*det_err[4])/hr**2
    # diff = np.real(det_err[2]-2*det_err[1]+det_err[0])/(hr**2) + 1j*np.imag(det_err[2]-2*det_err[1]+det_err[0])/(hr**2) # Accuracy 2, set N=3
    return diff

def d_alpha_omega(omega,alpha,m,Jet_parameters):
    N=5
    hr = 1e-6
    dD_err = np.zeros(N,dtype='complex')
    near_omega = np.linspace(np.real(omega)-2*hr,np.real(omega)+2*hr,N)+1j*np.imag(omega)
    for i in range(N):
        dD_err[i] = complex_diff(near_omega[i],alpha,m,Jet_parameters)
    diff = np.real(1/12*dD_err[0]-2/3*dD_err[1]+2/3*dD_err[3]-1/12*dD_err[4])/hr + 1j*np.imag(1/12*dD_err[0]-2/3*dD_err[1]+2/3*dD_err[3]-1/12*dD_err[4])/hr
    # diff = np.real(dD_err[2]-dD_err[0])/(2*hr) + 1j*np.imag(dD_err[2]-dD_err[0])/(2*hr)
    return diff

def build_jacobian(omega,alpha,m,Jet_parameters):
    
    dalpha, domega = complex_diff(omega,alpha,m,Jet_parameters, diffomega=True)
    d2alpha = dd_alpha(omega,alpha,m,Jet_parameters)
    d2omega = dd_omega(omega,alpha,m,Jet_parameters)
    dalphaomega = d_alpha_omega(omega,alpha,m,Jet_parameters)
    
    J = np.zeros((2,2),dtype='complex')
    J[0,0] = dalpha
    J[0,1] = domega
    J[1,0] = (d2alpha*domega-dalpha*dalphaomega)/domega**2
    J[1,1] = (dalphaomega*domega-dalpha*d2omega)/domega**2
    return J

def Newton_tracker(omega,alpha,m,Jet_parameters):
    # print(Jet_parameters.beta)
    roots = np.array([alpha,omega])
    search_root = True
    tol=1e-8
    iti=0
    while search_root:
        gv_err = group_velocity_calc(roots[1], roots[0], m, Jet_parameters)
        det_err = dispersion_relation(roots[1], roots[0], m, Jet_parameters)
        err_norm = np.linalg.norm(np.array([det_err,gv_err]))
        # print(f'iti = {iti}, err_norm = {err_norm}')
        J = build_jacobian(omega,alpha,m,Jet_parameters) 
        try:
            invJ = np.linalg.inv(J)
            roots = roots - invJ@[det_err, gv_err]
            if err_norm<tol: 
                search_root = False
                alpha= roots[0]
                omega= roots[1]
            elif iti>15: # No convergence
                search_root = False
                if err_norm>1e-1: 
                    print(f'Lost saddle, err = {err_norm}, beta = {Jet_parameters.beta}')
                    omega = np.NaN + 1j*np.nan # Saddle lost
                    alpha = np.NaN + 1j*np.nan
                else:
                    alpha= roots[0]
                    omega = roots[1]
        except Exception as e: 
            search_root = False
            omega = np.NaN + 1j*np.nan # Saddle lost
            alpha = np.NaN + 1j*np.nan
        iti +=1
    return omega, alpha

def mode_search(omega,m,Jet_parameters, ar_max=6,ai_min=-7,Nr=40,Ni=40):
    if np.real(omega)>4:
        ar_max = 1.5*np.real(omega)
    alpha_grid = build_grid(ar_max,0.001,0,ai_min,Nr,Ni)
    roots_raw = fsolve_opt(alpha_grid.flatten(),omega,m,Jet_parameters) 
    num_idx = ~np.isnan(roots_raw)
    roots_round = np.round(roots_raw[num_idx],decimals=5)
    roots_filtered = np.unique(roots_round, return_index=False)
    return roots_filtered

def build_grid(limRs, limRi, limIs, limIi, nr, ni):
    ar = np.linspace(limRi,limRs,nr)
    ai = np.linspace(limIi,limIs,ni)*1j
    arr,aii = np.meshgrid(ar,ai)
    return arr+aii

def filter_array(a,b):
    num_idx = ~np.isnan(a)
    num_idx2 = ~np.isnan(b)
    if (num_idx==num_idx2).all():
        print('Ok')
    else:
        print('Alpha and omega NaN are different')
    round_a = np.round(a[num_idx],decimals=5)
    round_b = np.round(b[num_idx],decimals=5)
    filtered_a, unique_idx = np.unique(round_a, return_index=True)
   
    return filtered_a, round_b[unique_idx]

###### NEW FUNCTIONS ######
###### _____________ ######

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