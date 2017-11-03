#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "pgmio.h"

#define P 4

#define NUM_ITER 100

int main(int argc, char *argv[]) {
  // MPI variables
  MPI_Comm comm;
  MPI_Status status;
  int size, rank, master_process = 0, next, prev;
  
  // Other variables
  char *file_name;
  int M, N, MP, NP;
  int i, j, iter;
  
  file_name = "edge192x128.pgm";
  pgmsize(file_name, &M, &N);
  
  MP = M/P;
  NP = N;

  float master_buffer[M][N];
  float buffer[MP][NP];
  float old[MP+2][NP+2];
  float new[MP+2][NP+2];
  float edge[MP+2][NP+2];
  
  // Initialise MPI and compute number of processes and local rank
  comm = MPI_COMM_WORLD;
  
  MPI_Init(&argc, &argv);
  
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  
  // Check if number of processes is correct
  if (P != size) {
    if (rank == master_process) {
      printf("ERROR: P = %d, size = %d.\n", P, size);
    }
    MPI_Finalize();
    exit(-1);
  }
  
  // Set neighbours' ranks
  next = rank + 1;
  prev = rank - 1;
  
  if (next >= size) {
    next = MPI_PROC_NULL;
  }
  if (prev < 0) {
    prev = MPI_PROC_NULL;
  }
  
  // With master process, read the image data into master buffer
  if (rank == master_process) {
    printf("Processing %d x %d image on %d processes\n", M, N, P);
    printf("Number of iterations = %d\n", NUM_ITER);
    
    printf("Reading <%s>\n", file_name);
    pgmread(file_name, master_buffer, M, N);
  }
  
  // Split the master buffer data up amongst processes
  MPI_Scatter(master_buffer, MP*NP, MPI_FLOAT, 
              buffer, MP*NP, MPI_FLOAT, 
              master_process, comm);
  
  // Copy buffer data into edge adding a 1 pixel padding
  for (i=1; i<=MP; ++i) {
    for (j=1; j<=NP; ++j) {
      edge[i][j] = buffer[i-1][j-1];
    }
  }
  
  // Initialize old values with 255.0 (white)
  for (i=0; i<MP+2; ++i) {
    for (j=0; j<NP+2; ++j) {
      old[i][j] = 255.0; 
    }
  }
  
  // Main loop
  for (iter=0; iter<NUM_ITER; ++iter) {
    // Do halo swaps
    MPI_Sendrecv(&old[MP][1], NP, MPI_FLOAT, next, 1, 
                 &old[0][1],  NP, MPI_FLOAT, prev, 1, 
                 comm, &status);
    
    MPI_Sendrecv(&old[1][1],    NP, MPI_FLOAT, prev, 2, 
                 &old[MP+1][1], NP, MPI_FLOAT, next, 2, 
                 comm, &status);
                 
    // Calculate the new value
    for (i=1; i<=MP; ++i) {
      for (j=1; j<=NP; ++j) {
        new[i][j] = 0.25 * (old[i-1][j] + old[i+1][j] + 
                            old[i][j-1] + old[i][j+1] - edge[i][j]); 
      }
    }
    
    // Update the old value with the new value
    for (i=1; i<=MP; ++i) {
      for (j=1; j<=NP; ++j) {
        old[i][j] = new[i][j]; 
      }
    }
  }
  
  // Copy data back to buffer
  for (i=0; i<MP; ++i) {
    for (j=0; j<NP; ++j) {
      buffer[i][j] = old[i+1][j+1];
    }
  }
  
  // Gather all the data from the buffers back to master buffer
  MPI_Gather(buffer, MP*NP, MPI_FLOAT, 
             master_buffer, MP*NP, MPI_FLOAT, 
             master_process, comm);
  
  // With master process, write the result image
  if (rank == master_process) {
    file_name="result192x128.pgm";
    printf("Writing <%s>\n", file_name);
    pgmwrite(file_name, master_buffer, M, N);
  }
  
  // Finalize the MPI environment
  MPI_Finalize();
  
  return 0;
}