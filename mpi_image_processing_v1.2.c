#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#include "pgmio.h"

#define P 4

#define FREQ 100

#define DELTA 0.1

int main(int argc, char *argv[]) {
  // MPI variables
  MPI_Comm comm;
  MPI_Status status;
  MPI_Request request_next, request_prev;
  int size, rank, root = 0, next, prev;
  
  // Other variables
  char *file_name;
  int M, N, MP, NP;
  int i, j, iter;
  int i_start, i_end;
  float local_delta, global_delta;
  
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
    if (rank == root) {
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
  if (rank == root) {
    printf("Processing %d x %d image on %d processes\n", M, N, P);
    
    printf("Reading <%s>\n", file_name);
    pgmread(file_name, master_buffer, M, N);
  }
  
  // Split the master buffer data up amongst processes
  MPI_Scatter(master_buffer, MP*NP, MPI_FLOAT, 
              buffer, MP*NP, MPI_FLOAT, 
              root, comm);
  
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
  iter = 1;
  global_delta = 1;
  do {
    // Send halos
    MPI_Issend(&old[MP][1], NP, MPI_FLOAT, next, 1, comm, &request_next);
    MPI_Issend(&old[1][1],  NP, MPI_FLOAT, prev, 2, comm, &request_prev);
    
    // Calculate indexes for halo independent pixels
    if (rank == 0) {
      i_start = 1;
    } else {
      i_start = 2;
    }
    if (rank == size-1) {
      i_end = MP;
    } else {
      i_end = MP-1;
    }
    
    // Calculate the new value on halo independent pixels
    for (i=i_start; i<=i_end; ++i) {
      for (j=1; j<=NP; ++j) {
        new[i][j] = 0.25 * (old[i-1][j] + old[i+1][j] + 
                            old[i][j-1] + old[i][j+1] - edge[i][j]); 
      }
    }
    
    // Receive halos
    MPI_Recv(&old[0][1],    NP, MPI_FLOAT, prev, 1, comm, &status);
    MPI_Recv(&old[MP+1][1], NP, MPI_FLOAT, next, 2, comm, &status);
    
    // Calculate the new value on halo dependent pixels
    if (rank != 0) {
      i = 1;
      for (j=1; j<=NP; ++j) {
        new[i][j] = 0.25 * (old[i-1][j] + old[i+1][j] + 
                            old[i][j-1] + old[i][j+1] - edge[i][j]); 
      }
    }
    if (rank != size-1) {
      i = MP;
      for (j=1; j<=NP; ++j) {
        new[i][j] = 0.25 * (old[i-1][j] + old[i+1][j] + 
                            old[i][j-1] + old[i][j+1] - edge[i][j]); 
      }
    }
    
    // Calculate the delta value
    if (iter%FREQ==0) {
      local_delta = 0;
      
      for (i=1; i<MP+1; ++i) {
        for (j=1; j<NP+1; ++j) {
          local_delta = fmax(local_delta, fabs(new[i][j] - old[i][j]));
        }
      }
      
      MPI_Allreduce(&local_delta, &global_delta, 1, MPI_FLOAT, MPI_MAX, comm);
    }
    
    // Update the old value with the new value
    for (i=1; i<=MP; ++i) {
      for (j=1; j<=NP; ++j) {
        old[i][j] = new[i][j]; 
      }
    }
    
    iter++;
  } while (global_delta >= DELTA);
  
  // Copy data back to buffer
  for (i=0; i<MP; ++i) {
    for (j=0; j<NP; ++j) {
      buffer[i][j] = old[i+1][j+1];
    }
  }
  
  // Gather all the data from the buffers back to master buffer
  MPI_Gather(buffer, MP*NP, MPI_FLOAT, 
             master_buffer, MP*NP, MPI_FLOAT, 
             root, comm);
  
  // With master process, write the result image
  if (rank == root) {
    printf("Finished with %d iterations\n", iter);
    
    file_name="result192x128.pgm";
    printf("Writing <%s>\n", file_name);
    pgmwrite(file_name, master_buffer, M, N);
  }
  
  // Finalize the MPI environment
  MPI_Finalize();
  
  return 0;
}