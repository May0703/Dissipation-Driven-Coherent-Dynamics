# -*- encoding: utf-8 -*-
#Desciption: dissipation & free evolution
#authur: Yue Wu
#Date: 2024/6/14
#Email:y-wu19@mails.tsinghua.edu.cn

using DifferentialEquations
using FFTW
using LinearAlgebra
using LaTeXStrings
using Printf
using HDF5
using GR
using DataFrames
using CSV

#-----------------------import data----------------------------

for i_omega in 5:9
    omega = 40 + 5*(i_omega-1)
    Zmax = 10.0     # Grid half length
    Npoint = 128    # Number of grid points
    g_s = 0.025/(omega)^0.5       # interaction strenth 
    whoz = 1.0       # harmonic oscilator angular frequency
    modecut = 20
    beta = 0.5*omega/2083 #200nK
    gamma1 = 2
    gamma2 = 1
    tspandis = 0.37
    tspan = omega/4

    #-----------------------derived data---------------------------

    Dz = 2*Zmax/Npoint              # length step size
    Dk = pi/Zmax            # momentum step size
    Kmax = Dk*div(Npoint,2)   # maximum momentum
    NormWF = 1.0/(2*Zmax)           # Normalization factor for Fourier basis

    #-----------------------grid definition------------------------  

    # spatial grid
    grid_z = -div(Npoint,2):div(Npoint,2)-1
    z = Dz * grid_z
    zp = fftshift(z); # for FFT order: Swap the first and second halves of the vector 
    #
    # momentum grid
    kp = Dk * grid_z
    kp = fftshift(kp);

    #-----------------------initial state--------------------------------------------
    fid = h5open("$(omega)_$(Npoint)", "r")
    phi = vec(fid["dataset_1"][:])
    u = vec(fid["dataset_2"][:])
    v = vec(fid["dataset_3"][:])
    ext = vec(fid["dataset_4"][:])
    mu = vec(fid["dataset_5"][:])[1]
    close(fid)

    function cn(_vec)
        _nex = zeros(Npoint,Npoint)*im
        for j=1:Npoint
            for k=1:Npoint
                for i=1:modecut
                    _nex[j,k] = _nex[j,k] + real(1 / (exp(beta*ext[i])-1)) * _vec[2*i,j]*_vec[2*i,k] + real(1 / (exp(beta*ext[i])-1)+1) * _vec[2*i+1,j]*_vec[2*i+1,k]
                end
            end
        end
        return _nex
    end

    function cns(vec)
        _ns = zeros(Npoint,Npoint)*im
        for j=1:Npoint
            for k=1:Npoint
                for i=1:modecut
                    _ns[j,k] = _ns[j,k] + (1 / (exp(beta*ext[i])-1))*(vec[2*i,j])*(-conj(vec[2*i+1,k])) + (1 / (exp(beta*ext[i])-1)+1)*(vec[2*i,k])*(-conj(vec[2*i+1,j]))
                end
            end
        end
        return _ns
    end

    vec0 = zeros(2*modecut+1,Npoint) * (1+im)
    vec0[1,:] = fftshift(phi)
    for i = 1:modecut
        vec0[2*i,:] = fftshift(u[(i-1)*Npoint+1:i*Npoint])
        vec0[2*i+1,:] = fftshift(v[(i-1)*Npoint+1:i*Npoint])
    end

    rvec0 = zeros(3,Npoint,Npoint) * (1+im)
    rvec0[1,1,:] = vec0[1,:]
    rvec0[2,:,:] = cn(vec0)
    rvec0[3,:,:] = cns(vec0)

    #------------------------define operators-------------------------

    # Kinetic energy in K space:
    Ekin_K = 0.5* kp.^2 
    # Potential energy in R space:
    Vpot_R = 0.5*whoz^2*zp.^2;  # Harmonic oscillator of angular frequency whoz

    #------------------------main functions-----------------------------

    function Hamiltonian_phi(_phi,_nex,_ns,s) 
        phi = _phi[1,:]
        nex = zeros(Npoint)*im
        ns = zeros(Npoint)*im
        for i=1:Npoint
            nex[i] = _nex[i,i]
            ns[i] = _ns[i,i]
        end
        Hu_position = Array{Complex{Float64}}(undef,Npoint,Npoint) # auxiliary vectors
        Hu_momentum = Array{Complex{Float64}}(undef,Npoint,Npoint)
        #
        Hu_position =@.((Vpot_R - mu + g_s * abs2(phi) + 2 * g_s * nex) * phi + g_s *ns *conj(phi))  # evolve u in configuration space
        u_k = fft(phi) # transform u to momentum space to evolve with Ekin
        Hu_momentum =  ifft(Ekin_K .* u_k) # evolve and transform back to configuration space 
        Ham_phi = zeros(Npoint,Npoint)*im
        Ham_phi[1,:] = -im * (Hu_momentum .+ Hu_position) - s*gamma1*phi
        return Ham_phi
    end

    function Hamiltonian_nex(_phi,_nex,_ns,s)
        phi = _phi[1,:]
        nex = zeros(Npoint)*im
        ns = zeros(Npoint)*im
        for i=1:Npoint
            nex[i] = _nex[i,i]
            ns[i] = _ns[i,i]
        end
        Hu_position = zeros(Npoint,Npoint)*im
        Hu_momentum = zeros(Npoint,Npoint)*im
        #
        for i=1:Npoint
            Hu_position[i,:] += @.((-mu + Vpot_R + 2*g_s*abs2.(phi) + 2*g_s*nex)*_nex[i,:] + (g_s*phi^2+g_s*ns)*conj(_ns[i,:]))
            Hu_position[:,i] += @.((mu - Vpot_R - 2*g_s*abs2.(phi) - 2*g_s*nex)*_nex[:,i] - (g_s*conj(phi)^2+g_s*conj(ns))*_ns[i,:])
            u_k = fft(_nex[i,:])
            Hu_momentum[i,:] += ifft(Ekin_K .* u_k) 
            u_k = fft(_nex[:,i])
            Hu_momentum[:,i] += -ifft(Ekin_K .* u_k) 
        end
        return  -im *(Hu_momentum .+ Hu_position) - s*gamma2*_nex
    end

    function Hamiltonian_ns(_phi,_nex,_ns,s)
        phi = _phi[1,:]
        nex = zeros(Npoint)*im
        ns = zeros(Npoint)*im
        for i=1:Npoint
            nex[i] = _nex[i,i]
            ns[i] = _ns[i,i]
        end
        Hu_position = zeros(Npoint,Npoint)*im
        Hu_momentum = zeros(Npoint,Npoint)*im
        #
        for i=1:Npoint
            Hu_position[i,:] += @.((-mu + Vpot_R + 2*g_s*abs2.(phi) + 2*g_s*nex)*_ns[i,:] + (g_s*phi^2+g_s*ns)*_nex[:,i] - im*1/2*s*gamma2*_ns[i,:] - im*1/2*s*gamma2*_ns[:,i])
            Hu_position[:,i] += @.((-mu + Vpot_R + 2*g_s*abs2.(phi) + 2*g_s*nex)*_ns[:,i] + (g_s*phi^2+g_s*ns)*_nex[:,i])
            u_k = fft(_ns[i,:])
            Hu_momentum[i,:] += ifft(Ekin_K .* u_k) 
            u_k = fft(_ns[:,i])
            Hu_momentum[:,i] += ifft(Ekin_K .* u_k) 
        end
        #Hu_ = zeros(Npoint)*im
        #for i=1:Npoint
        #    Hu_[i] = Hu_momentum[i,i]
        #end
        #plot(zp,real.(Hu_))
        #savefig("Hu_Hu2.png")
        return  -im *(Hu_momentum .+ Hu_position)
    end

    #=
    plot(zp,imag.(Hamiltonian_phi(rvec0[1,:,:],rvec0[2,:,:],rvec0[3,:,:])[1,:]))
    savefig("Hphi.png")

    _Hnex = Hamiltonian_nex(rvec0[1,:,:],rvec0[2,:,:],rvec0[3,:,:])
    _Hns = Hamiltonian_ns(rvec0[1,:,:],rvec0[2,:,:],rvec0[3,:,:])
    Hnex = zeros(Npoint)*im
    Hns = zeros(Npoint)*im
    for i=1:Npoint
        Hnex[i] = _Hnex[1,i]
        Hns[i] = _Hns[1,i]
    end
    plot(zp,imag.(Hnex))
    savefig("Hnex.png")
    plot(zp,imag.(Hns))
    savefig("Hns.png")
    =#

    function GPevolutiondis(dvec, vec, p, t)
        dvec[1,:,:] = Hamiltonian_phi(vec[1,:,:],vec[2,:,:],vec[3,:,:],1)
        dvec[2,:,:] = Hamiltonian_nex(vec[1,:,:],vec[2,:,:],vec[3,:,:],1)
        dvec[3,:,:] = Hamiltonian_ns(vec[1,:,:],vec[2,:,:],vec[3,:,:],1)
    end

    function GPevolution(dvec, vec, p, t)
        dvec[1,:,:] = Hamiltonian_phi(vec[1,:,:],vec[2,:,:],vec[3,:,:],0)
        dvec[2,:,:] = Hamiltonian_nex(vec[1,:,:],vec[2,:,:],vec[3,:,:],0)
        dvec[3,:,:] = Hamiltonian_ns(vec[1,:,:],vec[2,:,:],vec[3,:,:],0)
    end
    #-----------------------------------ODE------------------------------------------

    prob = ODEProblem(GPevolutiondis, rvec0, tspandis) # set the problem: Hamiltonian, initial state, and time range
    sol = solve(prob,saveat=0.001) #BS3() is the integrator, you can check for other integrators in the doc

    # print(sol.t)
    Nex_t = Array{Complex{Float64}}(undef,size(sol.t))
    N0_t = Array{Complex{Float64}}(undef,size(sol.t))
    for i = 1:size(sol.t)[1]
        soli = sol.u[i]
        nex = soli[2,:,:]
        nexd = zeros(Npoint)*im
        for j = 1:Npoint
            nexd[j] = nex[j,j]
        end
        global Nex_t[i] = Dz * sum(nexd)
        global N0_t[i] = Dz * sum(abs2.(soli[1,:,:]))
    end
    Nex_t = convert(Array{Float64}, real.(Nex_t))
    N0_t = convert(Array{Float64}, real.(N0_t))
    print(Nex_t[1])
    print(",")

    legend(L"N_e/N", L"N_0/N")
    xlabel(L"t/ms")
    ylabel("fraction")
    plot((sol.t)*(1000/(2*3.14159*omega)), Nex_t,"r",(sol.t)*(1000/(2*3.14159*omega)), N0_t,"g")
    savefig("$(omega)dis.png")

    df = DataFrame(x=N0_t)
    CSV.write("$(omega)dis_0.csv",df)
    df = DataFrame(x=Nex_t)
    CSV.write("$(omega)dis_e.csv",df)

    rvec0 = sol.u[end]

    prob = ODEProblem(GPevolution, rvec0, tspan) # set the problem: Hamiltonian, initial state, and time range
    sol = solve(prob,saveat=0.1) #BS3() is the integrator, you can check for other integrators in the doc
    Nex_t = Array{Complex{Float64}}(undef,size(sol.t))
    N0_t = Array{Complex{Float64}}(undef,size(sol.t))
    for i = 1:size(sol.t)[1]
        soli = sol.u[i]
        nex = soli[2,:,:]
        nexd = zeros(Npoint)*im
        for j = 1:Npoint
            nexd[j] = nex[j,j]
        end
        global Nex_t[i] = Dz * sum(nexd)
        global N0_t[i] = Dz * sum(abs2.(soli[1,:,:]))
    end
    Nex_t = convert(Array{Float64}, real.(Nex_t))
    N0_t = convert(Array{Float64}, real.(N0_t))
    print(Nex_t[1])
    print(",")

    print(sol.t)

    legend(L"N_e/N", L"N_0/N")
    xlabel(L"t/ms")
    ylabel("fraction")
    plot((sol.t)*(1000/(2*3.14159*omega)), Nex_t,"r",(sol.t)*(1000/(2*3.14159*omega)), N0_t,"g")
    savefig("$(omega).png")

    df = DataFrame(x=N0_t)
    CSV.write("$(omega)_0.csv",df)
    df = DataFrame(x=Nex_t)
    CSV.write("$(omega)_e.csv",df)
end