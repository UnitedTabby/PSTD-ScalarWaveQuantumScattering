MPIICPX=mpiicpx -std=c++17
SCATFLAGS=-xALDERLAKE -O3 -qopenmp -qopt-report -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -liomp5 -lpthread
QSCATTER=PSTD_QScat.cpp potential.cpp utility.cpp qscat.cpp
HEADER_QSCATTER=PSTD_QScat.h run_environment.h utility.h
num_threads?=1

default: float double

fqscat:	$(QSCATTER) $(HEADER_QSCATTER)
	echo =======================make fqscat=======================
	$(MPIICPX) $(SCATFLAGS) -D__USE_FLOAT__ -D__OMP_NUM_THREADS__=$(num_threads) -D$(show_init_status) -D$(smooth_profile) -o fqscat $(QSCATTER) -lm

dqscat:	$(QSCATTER) $(HEADER_QSCATTER)
	echo =======================make dqscat=======================
	$(MPIICPX) $(SCATFLAGS) -D__OMP_NUM_THREADS__=$(num_threads) -D$(show_init_status) -D$(smooth_profile) -o dqscat $(QSCATTER) -lm

clean:
	rm -f fqscat dqscat *.optrpt *.opt.yaml
