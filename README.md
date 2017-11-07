# MPI_ImageProcessing
Simple implementation of a MPI parallel program for very basic image processing algorithm with C and MPI. The input are ﬁles representing the output from a very simple edge-detection algorithm applied to a greyscale image. The objective is to do the reverse operation and construct the initial image given the edges.

Compile with: 

      > gcc mpi_image_processing.c pgmio.h pgmio.c -lmsmpi -o mpi_image_processing.exe

Run with: 

      > mpiexec -n 4 mpi_image_processing.exe

where `-n x` sets the number of processes

Setup for windows:
1. Install MinGW
2. Install MS-MPI
3. Adapt MS-MPI for MinGW
  - Create the `libmsmpi64.a` library with the MinGW64 tools `gendef` and `dlltool`
      ```
      > gendef msmpi.dll                                  # generate msmpi.def
      > dlltool -d msmpi.def -D msmpi.dll -l libmsmpi.a   # generate the (static) library file libmsmpi.a
      ```
  - Copy the new library to where g++ looks for them, e.g. `/mingw64/lib`
  - Modify the header file `mpi.h`. Add `#include <stdint.h>` above `typedef __int64 MPI_Aint`
  - Copy the modified header file `mpi.h` to the default include folder e.g. `/mingw64/include`


For more details see:
- http://www.math.ucla.edu/~wotaoyin/windows_coding.html
- https://github.com/coderefinery/autocmake/issues/85 
