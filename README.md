# PSTD-ScalarWaveQuantumScattering
The software is a parallel PSTD solver to the quantum scattering of a monochromatic plane wave by a potential of arbitary form.  It employs an absorbing boundary condition for lattice truncation, a total-field/scattered-field approach for the incident wave source 
condition, and the method of pseudo-spectral time-domain on local Fourier basis for solving the Schrodinger equation of scalar wavefunction, corresponding to the scattering of scalar particles.  There is no restriction on the function form of the potential, except it has finite range.  Overlapping domain decomposition is employed and the Fourier transform on the local data makes the PSTD parallelizable.  Details about the theoretical derivations and the parallel algorithm can be found in the paper arXiv:2403.04053, https://doi.org/10.48550/arXiv.2403.04053.

This code requires the Intel OneAPI with MPI and MKL libraries, which enables the MPI-OpenMP-SIMD-vectorization hybrid parallel execution. The output are the grid values of the wave function and its surface normal derivative on the virtual surface, stored in a file, with the filename defined in the input file.  An example input file is provided.  On a workstation of two CPUs, typical commands are,

for compiling:

  num_threads=10 show_init_status=__INIT_STATUS__ make dqscat
  
for execution:

  mpirun -np 2 -iface lo ./dqscat -i InputExample.txt

The source code is copyrighted by: Kun Chen, Shanghai Institute of Optics and Fine Mechanics, Chinese Academy of Sciences

and it is distributed under the terms of the MIT license.

Please see LICENSE file for details.
