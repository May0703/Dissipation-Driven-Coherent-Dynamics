# -*- encoding: utf-8 -*-
#Desciption: dissipation & free evolution & dissipation
#authur: Yue Wu
#Date: 2024/6/14
#Email:y-wu23@mails.tsinghua.edu.cn

using DifferentialEquations
using FFTW
using LinearAlgebra
using LaTeXStrings
using Printf
using HDF5
using GR
using DataFrames
using CSV
using JLD

#-----------------------import data----------------------------

i_omega = 9
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
point = 80
tspan = omega/6/point

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
    return  -im *(Hu_momentum .+ Hu_position)
end

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
Nex_t = Array{Complex{Float64}}(undef,point+1)
N0_t = Array{Complex{Float64}}(undef,point+1)

prob = ODEProblem(GPevolutiondis, rvec0, tspandis*2) # set the problem: Hamiltonian, initial state, and time range
sol = solve(prob) #BS3() is the integrator, you can check for other integrators in the doc
rvec0 = sol.u[end]
nex = rvec0[2,:,:]
nexd = zeros(Npoint)*im
for j = 1:Npoint
    nexd[j] = nex[j,j]
end
Nex_t[1] = Dz * sum(nexd)
N0_t[1] = Dz * sum(abs2.(rvec0[1,:,:]))
print(Nex_t[1]+N0_t[1])

save("80afterdis.jld", "x", rvec0)
rvec0 = load("80afterdis.jld", "x")


for i in 1:point
    print(i)
    prob = ODEProblem(GPevolution, rvec0, tspan) # set the problem: Hamiltonian, initial state, and time range
    sol = solve(prob) #BS3() is the integrator, you can check for other integrators in the doc
    global rvec0 = sol.u[end]
    save("80_$(i).jld", "x", rvec0)
    nex = rvec0[2,:,:]
    nexd = zeros(Npoint)*im
    for j = 1:Npoint
        nexd[j] = nex[j,j]
    end
    global Nex_t[i+1] = Dz * sum(nexd)
    global N0_t[i+1] = Dz * sum(abs2.(rvec0[1,:,:]))
end

for i in 1:point
    print(i)
    rvec0 = load("80_$(i).jld", "x")
    prob = ODEProblem(GPevolutiondis, rvec0, tspandis) # set the problem: Hamiltonian, initial state, and time range
    sol = solve(prob) #BS3() is the integrator, you can check for other integrators in the doc
    rvec0 = sol.u[end]
    save("80_$(i)dis.jld", "x", rvec0)
    nex = rvec0[2,:,:]
    nexd = zeros(Npoint)*im
    for j = 1:Npoint
        nexd[j] = nex[j,j]
    end
    global Nex_t[i+1] = Dz * sum(nexd)
    global N0_t[i+1] = Dz * sum(abs2.(rvec0[1,:,:]))
end

Nex_t = convert(Array{Float64}, real.(Nex_t))
N0_t = convert(Array{Float64}, real.(N0_t))
N_t = Nex_t .+ N0_t

df = DataFrame(x=N_t)
CSV.write("Nt.csv",df)
