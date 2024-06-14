# -*- encoding: utf-8 -*-
'''
@Desciption: Generating the wavefunction of condensate and thermal parts in equilibrium
@authur: Yue Wu
@Date: 2024/6/14
@Email:y-wu23@mails.tsinghua.edu.cn
'''

import numpy as np
from numpy.linalg import eig, eigh
import h5py
from scipy.integrate import solve_ivp

L = 10
Ngrid = 512
dx = L*2/Ngrid
x = np.arange(-L,L+dx,dx)

def equilibrium(omega,g,beta,modecut,rate):
#-------------------------------function-----------------------------------
    
    def n0_r(_phi,_N0): 
        '''
        calculate density of condensate
        '''
        return _N0 * np.abs(_phi)**2

    def norm(_uv):
        for i in range(modecut):
            _norm = 0
            for j in range(Ngrid+1):
                _norm = _norm + (np.abs(_uv[i][j])**2 - np.abs(_uv[i][Ngrid+1+j])**2)*dx
            _uv[i] = _uv[i]/(_norm)**0.5
        return _uv
    def nex_r(_uv,_ext):
        '''
        calculate density of thermal atoms
        '''
        ni_r = np.zeros((modecut,Ngrid+1)) #|u|^2+|v|^2
        for i in range(modecut):
            for j in range(Ngrid+1):
                ni_r[i,j] = np.abs(_uv[i][j])**2 + np.abs(_uv[i][Ngrid+1+j])**2
        return np.dot(np.transpose(1/(np.exp(_ext*beta)-1)), ni_r)
    def mm_r(_uv,_ext):
        '''
        calculate anomolous density of thermal atoms
        '''
        ni_r = np.zeros((modecut,Ngrid+1)) * 0j
        for i in range(modecut):
            for j in range(Ngrid+1):
                ni_r[i,j] = _uv[i][j]*(np.conjugate(_uv[i][Ngrid+1+j])) #-uv*
        return np.dot(np.transpose(2*1/(np.exp(_ext*beta)-1)+1), ni_r)
    
    #--------------------------------zero-------------------------------------

    H0 = np.diag(1/2*x**2+(-2)*(-1/(2*dx**2))+0j)
    for i in range(0,Ngrid): #subdiagonal
        H0[i][i+1] = 1*(-1/(2*dx**2))
        H0[i+1][i] = 1*(-1/(2*dx**2))


    eigval, eigvec = eigh(H0)
    mu = eigval[0]
    ext_all = eigval[1:]-eigval[0]


    ext = ext_all[:modecut]

    phi0 = np.transpose(eigvec)[0] / np.sqrt(dx)
    u = np.transpose(eigvec)[1:(modecut+1)]
    uv = np.zeros((modecut,(Ngrid+1)*2),dtype=np.complex128)
    uv[:,:(Ngrid+1)] = u
    uv = norm(uv)

    N_ex = np.sum(dx * nex_r(uv,ext))
    N = rate*N_ex
    N0 = N - N_ex
    l_n0 = _n0 = n0_r(phi0,N0)
    l_nex = _nex = nex_r(uv,ext)
    l_mm = _mm = mm_r(uv,ext)

    #-------------------------------iterate----------------------------------
    
    l_nex = _nex
    diff = 100
    turn=3

    while(1):
        def evo_phi(t,phi):
            H_phi = H0 + g*np.diag(n0_r(phi/np.sqrt(np.dot(np.transpose(np.conjugate(phi)),phi)*dx),N0))+ 2*g*np.diag(_nex) + g*np.diag(_mm)
            return - np.dot(H_phi,phi)

        sol = solve_ivp(evo_phi,(0,1),phi0)

        tsol = sol.t
        ysol = np.transpose(sol.y)
        E = np.zeros(np.shape(tsol)[0])
        for i in range(np.shape(tsol)[0]):
            E[i] = np.dot(np.dot(np.transpose(np.conjugate(ysol[i])),H0 +g*np.diag(n0_r(ysol[i]/np.sqrt(np.dot(np.transpose(np.conjugate(ysol[i])),ysol[i])*dx),N0))+ 2*g*np.diag(_nex)+ g*np.diag(_mm)),ysol[i]) / np.dot(np.transpose(np.conjugate(ysol[i])),ysol[i])

        phi = ysol[-1] / np.sqrt(np.dot(np.transpose(np.conjugate(ysol[-1])),ysol[-1])*dx)
        while(1):
            sol = solve_ivp(evo_phi,(0,0.1),phi)
            tsol = sol.t
            ysol = np.transpose(sol.y)
            diffE = np.abs(np.dot(np.dot(np.transpose(np.conjugate(ysol[-1])),H0 +g*np.diag(n0_r(ysol[-1]/np.sqrt(np.dot(np.transpose(np.conjugate(ysol[-1])),ysol[-1])*dx),N0))+ 2*g*np.diag(_nex)+ g*np.diag(_mm)),ysol[-1]) / np.dot(np.transpose(np.conjugate(ysol[-1])),ysol[-1]) - np.dot(np.dot(np.transpose(np.conjugate(ysol[0])),H0 +g*np.diag(n0_r(ysol[0]/np.sqrt(np.dot(np.transpose(np.conjugate(ysol[0])),ysol[0])*dx),N0))+ 2*g*np.diag(_nex)+ g*np.diag(_mm)),ysol[0]) / np.dot(np.transpose(np.conjugate(ysol[0])),ysol[0]))
            phi = ysol[-1] / np.sqrt(np.dot(np.transpose(np.conjugate(ysol[-1])),ysol[-1])*dx)
            if diffE < 0.00001:
                break
        
        mu = np.dot(np.dot(np.transpose(np.conjugate(ysol[-1])),H0 +g*np.diag(n0_r(ysol[-1]/np.sqrt(np.dot(np.transpose(np.conjugate(ysol[-1])),ysol[-1])*dx),N0))+ 2*g*np.diag(_nex)+ g*np.diag(_mm)),ysol[-1]) / np.dot(np.transpose(np.conjugate(ysol[-1])),ysol[-1])
        l_n0 = _n0 = n0_r(phi,N0)

        turn = 0
        diff = 100
        while(diff > 0.05):
            H_uv = np.zeros(((Ngrid+1)*2,(Ngrid+1)*2),dtype=np.complex128)
            H_uv[:(Ngrid+1),:(Ngrid+1)] = H0 + 2*g*np.diag(l_n0) + 2*g*np.diag(l_nex) - np.identity(Ngrid+1)*mu
            H_uv[(Ngrid+1):,(Ngrid+1):] = - (H0 +2*g*np.diag(l_n0) + 2*g*np.diag(l_nex) - np.identity(Ngrid+1)*mu)
            H_uv[:(Ngrid+1),(Ngrid+1):] = g*np.diag(l_n0) + g*np.diag(l_mm)
            H_uv[(Ngrid+1):,:(Ngrid+1)] = -g*np.diag(l_n0) - g*np.diag(l_mm)
            eigval, eigvec = eig(H_uv)
            idx = eigval.argsort()  
            eigval = eigval[idx]
            ext = eigval[Ngrid+2:Ngrid+2+modecut]
            uv = np.transpose(eigvec[:,idx])[Ngrid+2:Ngrid+2+modecut]
            uv = norm(uv)
            _nex = nex_r(uv,ext)
            _mm = mm_r(uv,ext)
            N_ex = np.sum(dx * _nex)
            N0 = N - N_ex
            diff = np.sum(np.abs(l_nex-_nex))
            print(diff)
            l_n0 = (_n0+l_n0)/2
            l_nex = (_nex+l_nex)/2
            l_mm = (_mm+l_mm)/2
            turn = turn + 1
        if turn == 1:
            break

    Ngrid_ = 128
    _phi = np.zeros(Ngrid_)*(1+0j)
    _u = np.zeros((modecut,Ngrid_))*(1+0j)
    _v = np.zeros((modecut,Ngrid_))*(1+0j)
    for i in range(Ngrid_):
        _phi[i] = np.sqrt(N0)*phi[4*i]
        _u[:,i] = uv[:,4*i]
        _v[:,i] = -uv[:,Ngrid+1+4*i]

    _u = np.reshape(_u,modecut*Ngrid_)
    _v = np.reshape(_v,modecut*Ngrid_)

    hdfFile = h5py.File(f"{omega}_{Ngrid_}", "w")
    hdfFile.create_dataset('dataset_1', data=_phi)
    hdfFile.create_dataset('dataset_2', data=_u)
    hdfFile.create_dataset('dataset_3', data=_v)
    hdfFile.create_dataset('dataset_4', data=ext)
    hdfFile.create_dataset('dataset_5', data=np.array([mu,mu]))
    hdfFile.close()

    return 0

for _omega in range(40,85,5):
    _g = 0.025/(_omega)**0.5 
    _beta = 0.5*_omega/2083  
    _modecut = 20
    _rate = 4.3
    equilibrium(_omega,_g,_beta,_modecut,_rate)
