/*
 * MIT License
 *
 *
 * Copyright (c) 2025 Kun Chen <kunchen@siom.ac.cn>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <string.h>
#include <mpi.h>
#include <sys/time.h>
#include "run_environment.h"
#include "PSTD_QScat.h"
#include "utility.h"

using namespace std;

void show_usage(char *app);
void load_data(string& openname, int &nx_proc, int &ny_proc, int &nz_proc, QPrecision &dtau,
      QPrecision &dx, QPrecision &dy, QPrecision &dz, long &nx_fft, long &ny_fft, long &nz_fft,
      long &n_abc, long &n_sca, long &n_ksi, long &n_halo, QPrecision &theta, QPrecision &phi,
      QPrecision &E0, QPrecision &CAP_U0, QPrecision &CAP_alpha, long &Iter, string& surfname_prefix);

int main(int argc, char *argv[])
{
   /* the number of threads per processor will be set via the num_threads environment
    * variable on the make command-line;
    */

   /* read in parameters from input file */
   /* get the numbers of processors nx_proc, ny_proc, nz_proc
    * in x, y, z directions;
    *
    * get dtau, dx, dy, dz in unit of radian;
    * 
    * get nx_fft, ny_fft, nz_fft;
    * (for the sake of FFT performance, these are carefully chosen beforehand)
    * (the total number of grids in the model, nx_glb, ny_glb, nz_glb are derived)
    * 
    * get the thicknesses of abc, scattered-field region, ksi, domain overlapping;
    * (the positions of ksi region (the transition layer), virtual surfaces are derived)
    *
    * get the direction (theta,phi) and energy of the incident wave;
    *
    * get the ABC boundary CAP_U0 and CAP_alpha;
    *
    * get the number of total iterations to run;
    *
    * get the file name to save the surface terms;
    *
    * Error code: 12, i.e., ENOMEM, memory allocation error, insufficient memory;
    * 		  22, i.e., EINVAL, memory allocation error, invalid argument;
    * 		  2, the 3D process topology mismatches the total number of processes
    * 		  3, file I/O error;
    * 		  4, incomplete file for model parameters
    */

   string openname;
   for (int i=1; i<argc; ++i) {
      string arg=argv[i];
      if ((arg=="-h") || (arg=="--help")) {
	 show_usage(argv[0]);
	 return 0;
      }
      if (arg=="-i") {
	 if (i<argc-1) 
	    openname=argv[i+1];
	 if (i==argc-1 || openname[0]=='-') {
	    cerr << "Error: -i requires an argument" << endl;
	    show_usage(argv[0]);
	    return 1;
	 }
      }
   }

   if (openname.empty()) {
      cerr << "Error: mandatory -i FILE for model paramters" << endl;
      show_usage(argv[0]);
      return 1;
   }

   MPI_Init(&argc, &argv);

   try {
      int nx_proc, ny_proc, nz_proc;
      QPrecision dtau, dx, dy, dz;
      long nx_fft, ny_fft, nz_fft, n_abc, n_sca, n_ksi, n_halo;
      QPrecision theta, phi, E0, CAP_U0, CAP_alpha;
      long Iter;  // total iterations to run
      string surfname_prefix;  // filename to save the surface terms

      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank==0) {
	 load_data(openname, nx_proc, ny_proc, nz_proc,
	       dtau, dx, dy, dz, nx_fft, ny_fft, nz_fft,
	       n_abc, n_sca, n_ksi, n_halo,
	       theta, phi, E0, CAP_U0, CAP_alpha, Iter, surfname_prefix);
      }
      // MPI standard v3.0 section 5.1: no guarantee of synchronization
      // for collective communication routines.  apply barrier here
      MPI_Bcast(&nx_proc, 1, MPIInt, 0, MPI_COMM_WORLD);
      MPI_Bcast(&ny_proc, 1, MPIInt, 0, MPI_COMM_WORLD);
      MPI_Bcast(&nz_proc, 1, MPIInt, 0, MPI_COMM_WORLD);
      MPI_Bcast(&dtau, 1, MPIQPrecision, 0, MPI_COMM_WORLD);
      MPI_Bcast(&dx, 1, MPIQPrecision, 0, MPI_COMM_WORLD);
      MPI_Bcast(&dy, 1, MPIQPrecision, 0, MPI_COMM_WORLD);
      MPI_Bcast(&dz, 1, MPIQPrecision, 0, MPI_COMM_WORLD);
      MPI_Bcast(&nx_fft, 1, MPILong, 0, MPI_COMM_WORLD);
      MPI_Bcast(&ny_fft, 1, MPILong, 0, MPI_COMM_WORLD);
      MPI_Bcast(&nz_fft, 1, MPILong, 0, MPI_COMM_WORLD);
      MPI_Bcast(&n_abc, 1, MPILong, 0, MPI_COMM_WORLD);
      MPI_Bcast(&n_sca, 1, MPILong, 0, MPI_COMM_WORLD);
      MPI_Bcast(&n_ksi, 1, MPILong, 0, MPI_COMM_WORLD);
      MPI_Bcast(&n_halo, 1, MPILong, 0, MPI_COMM_WORLD);
      MPI_Bcast(&theta, 1, MPIQPrecision, 0, MPI_COMM_WORLD);
      MPI_Bcast(&phi, 1, MPIQPrecision, 0, MPI_COMM_WORLD);
      MPI_Bcast(&E0, 1, MPIQPrecision, 0, MPI_COMM_WORLD);
      MPI_Bcast(&CAP_U0, 1, MPIQPrecision, 0, MPI_COMM_WORLD);
      MPI_Bcast(&CAP_alpha, 1, MPIQPrecision, 0, MPI_COMM_WORLD);
      MPI_Bcast(&Iter, 1, MPI_LONG, 0, MPI_COMM_WORLD);
      int cnt;
      if (!rank) cnt=surfname_prefix.length();
      MPI_Bcast(&cnt, 1, MPIInt, 0, MPI_COMM_WORLD);
      char *str=new char[cnt+1];
      str[cnt]='\0';
      if (!rank) strcpy(str, surfname_prefix.c_str());
      MPI_Bcast(str, cnt, MPI_CHAR, 0, MPI_COMM_WORLD);
      if (rank) surfname_prefix=str;
      delete[] str;

      CPSTD_QScat pstd_qscat;
      pstd_qscat.Init(nx_proc, ny_proc, nz_proc, dtau, dx, dy, dz, nx_fft, ny_fft, nz_fft, n_abc,
	    n_sca, n_ksi, n_halo, theta, phi, E0, CAP_U0, CAP_alpha, V, MPI_COMM_WORLD);

      pstd_qscat.Save_g();
      pstd_qscat.SavePsi_XPlane("Init", 0.0);
      pstd_qscat.SavePsi_YPlane("Init", 0.0);
      pstd_qscat.SavePsi_ZPlane("Init", 0.0);

      double tstart=MPI_Wtime();
      long iter_accum_duration=pstd_qscat.IterationsToAccumulateSurfaceTerms();
      long iter_th;
      bool flgAccum=false;
      for (long i=1; i<=Iter; ++i) {
	 if ( !(rank || i%(Iter>>7)) )
	    cout << "Iter=" << i << ", clock time: " <<
	       MPI_Wtime()-tstart << endl;
	 
	 pstd_qscat.Update();
	 if (!(i%(Iter/10))) {
	    pstd_qscat.SavePsi_XPlane("Iter", 0.0);
	    pstd_qscat.SavePsi_YPlane("Iter", 0.0);
	    pstd_qscat.SavePsi_ZPlane("Iter", 0.0);
	 }

	 if (!flgAccum) {
	    if (i+iter_accum_duration-1>=Iter) {
	       iter_th=Iter;
	       flgAccum=true;
	    } else if (i+iter_accum_duration == 20001 || i+iter_accum_duration == 40001
		  || i+iter_accum_duration == 60001) {
	       iter_th=i+iter_accum_duration-1;
	       flgAccum=true;
	    }
	 }
	 if (flgAccum) {
	    pstd_qscat.AccumulateSurfaceTerms();
	    if (i==iter_th) {
	       pstd_qscat.SaveVirtualSurfaces(surfname_prefix);
	       flgAccum=false;
	    }
	 }
      }
   } catch (exception const &e) {
      cerr << e.what() << endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
   }

   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Finalize();
}

void show_usage(char *app)
{
   cout << "Usage: " << app << " -np # -i FILE\n" << \
      "where, #: number of processes; FILE: input file for model parameters" << endl;
}

void load_data(string& openname, int &nx_proc, int &ny_proc, int &nz_proc, QPrecision &dtau,
      QPrecision &dx, QPrecision &dy, QPrecision &dz, long &nx_fft, long &ny_fft, long &nz_fft,
      long &n_abc, long &n_sca, long &n_ksi, long &n_halo, QPrecision &theta, QPrecision &phi,
      QPrecision &E0, QPrecision &CAP_U0, QPrecision &CAP_alpha, long &Iter, string& surfname_prefix)
{
   std::ifstream ifs(openname.c_str());
   string aline;
   stringstream sstr;

   int count=0;

   if (!ifs.is_open())
      throw runtime_error("Error 3: Cannot open input file " + openname);

   if (getaline(ifs,aline)) {
      sstr.str(aline);
      sstr >> nx_proc >> ny_proc >> nz_proc;
      count++;
   }

   if (getaline(ifs,aline)) {
      sstr.clear();
      sstr.str(aline);
      sstr >> dtau >> dx >> dy >> dz;
      count++;
   }

   if (getaline(ifs,aline)) {
      sstr.clear();
      sstr.str(aline);
      sstr >> nx_fft >> ny_fft >> nz_fft;
      count++;
   }

   if (getaline(ifs,aline)) {
      sstr.clear();
      sstr.str(aline);
      sstr >> n_abc >> n_sca >> n_ksi >> n_halo;
      count++;
   }

   if (getaline(ifs,aline)) {
      sstr.clear();
      sstr.str(aline);
      sstr >> theta >> phi;
      count++;
   }

   if (getaline(ifs,aline)) {
      sstr.clear();
      sstr.str(aline);
      sstr >> E0;
      count++;
   }

   if (getaline(ifs,aline)) {
      sstr.clear();
      sstr.str(aline);
      sstr >> CAP_U0;
      count++;
   }

   if (getaline(ifs,aline)) {
      sstr.clear();
      sstr.str(aline);
      sstr >> CAP_alpha;
      count++;
   }

   if (getaline(ifs,aline)) {
      sstr.clear();
      sstr.str(aline);
      sstr >> Iter;
      count++;
   }

   if (getaline(ifs,aline)) {
      surfname_prefix=aline;
      count++;
   }

   if (count<10)
      throw runtime_error("Error 4: incomplete parameter file " + openname);
}
