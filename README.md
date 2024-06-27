Dissipation-Driven-Coherent-Dynamics

In these codes we simulate how thermal and condensate part of bosons in a 1D harmonic trap exchange atoms after dissipation. 

"initial.py" calculates the atom density in equilibruim. Imaginary time evolution and Iteration method are used to generated the wave function of condensate and quasi-particles respectively when interaction and trap potential both exist.

"evo_n_dis" and "evo_n_disdis" calculate the evolution of wave function of condesate and density matrix of thermal atoms. In "evo_n_dis", time sequence is "dissipation-evolution" but in "evo_n_disdis", time sequence is "dissipation-evolution-dissipation". Faster Fourier Transform algorithm is performed to accelarate the codes.
