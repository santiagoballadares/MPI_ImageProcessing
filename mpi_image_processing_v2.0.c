#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#include "pgmio.h"

#define FREQ 100

#define DELTA 0.1

int main(int argc, char *argv[]) {
  // MPI variables
  MPI_Comm comm, cart_comm;
  MPI_Status status;
  MPI_Request request_left, request_right, request_up, request_down;
  MPI_Datatype row, column, block;
  int size, rank, root = 0, tag = 0;
  int rank_left, rank_right, rank_up, rank_down;
  int tag_left = 1, tag_right = 2, tag_up = 3, tag_down = 4;
  int ndims, dims[2], periods[2], reorder;
  int coords[2];
  
  // Other variables
  char *file_name;
  int P, M, N, MP, NP;
  int i, j, iter;
  float local_delta, global_delta;
  
  // 2d topology
  ndims = 2;        // 2d grid
  dims[0] = 0;      // rows
  dims[1] = 0;      // columns
  periods[0] = 0;		// row-nonperiodic
  periods[1] = 0;		// column-nonperiodic
  reorder = 0;      // no rank reorder
  
  // Initialise MPI and compute number of processes and local rank
  comm = MPI_COMM_WORLD;
  
  MPI_Init(&argc, &argv);
  
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  
  P = size;
  
  // Distribute processes in a 2d topology
  if (MPI_Dims_create(P, ndims, dims) != 0) {
    if (rank == root) {
      printf("Error: Cannot distribute processes. P = %d, D = %d\n", P, ndims);
    }
    MPI_Finalize();
    exit(-1);
  }
  
  if (rank == root)
    printf("Topology: P=%d, ndims=%d, dims[0]=%d, dims[1]=%d\n", 
      P, ndims, dims[0], dims[1]);
  
  // Get image dimensions
  file_name = "edge192x128.pgm";
  pgmsize(file_name, &M, &N);
  
  MP = M / dims[1];
  NP = N / dims[0];
  
  // Split image dimmensions among processes
  if ((MP*dims[1] != M) || (NP*dims[0] != N)) {
    if (rank == root) {
      printf("Error: Cannot split image dimmensions among processes. \
              M=%d, (%d*%d)=%d; N=%d, (%d*%d)=%d\n", 
              M, MP, dims[1], MP*dims[1], 
              N, NP, dims[0], NP*dims[0]);
    }
    MPI_Finalize();
    exit(-1);
  }
  
  if (rank == root)
    printf("Image dimensions: M=%d, N=%d, MP=%d, NP=%d\n", M, N, MP, NP);
  
  // Buffers
  float master_buffer[M][N];
  float buffer[MP][NP];
  float old[MP+2][NP+2];
  float new[MP+2][NP+2];
  float edge[MP+2][NP+2];
  
  // Define derived data types
  MPI_Type_vector(MP, NP, N, MPI_FLOAT, &block);
  MPI_Type_commit(&block);
  
  MPI_Type_vector(1, NP, NP+2, MPI_FLOAT, &column);
  MPI_Type_commit(&column);
  
  MPI_Type_vector(MP, 1, NP+2, MPI_FLOAT, &row);
  MPI_Type_commit(&row);
  
  // Create virtual topology and set neighbours' ranks
  MPI_Cart_create(comm, ndims, dims, periods, reorder, &cart_comm);
  
  MPI_Comm_rank(cart_comm, &rank);
  
  MPI_Cart_shift(cart_comm, 1, 1, &rank_left, &rank_right);
  MPI_Cart_shift(cart_comm, 0, 1, &rank_up, &rank_down);
  
  /*printf("rank = %d: left = %d, right = %d, up = %d, down = %d\n", 
          rank, rank_left, rank_right, rank_up, rank_down);*/
  
  // With root process, read the image data into master buffer
  if (rank == root) {
    printf("Processing %d x %d image on %d processes\n", M, N, P);
    
    printf("Reading <%s>\n", file_name);
    pgmread(file_name, master_buffer, M, N);
  }
  
  // Split the master buffer data up amongst processes
  if (rank == root) {
    // Copy local data
    for (i=0; i<MP; ++i) {
      for (j=0; j<NP; ++j) {
        buffer[i][j] = master_buffer[i][N-NP+j];
      }
    }
    
    // Calculate block coordinates and send data to processes
    int x, y, p;
    for (p=1; p<P; ++p) {
      MPI_Cart_coords(cart_comm, p, ndims, coords);
      
      x =  M - (M - coords[1] * MP);
      y =  N - ((coords[0] + 1) * NP);
      
      /*printf("rank = %d: p = %d; coords[0] = %d, coords[1] = %d; (%d, %d)\n",
              rank, p, coords[0], coords[1], x, y);*/
    
      MPI_Ssend(&(master_buffer[x][y]), 1, block, p, tag, cart_comm);
    }
  } else {
    // Receive data
    MPI_Recv(&(buffer[0][0]), MP*NP, MPI_FLOAT, root, tag, cart_comm, &status);
  }
  
  // Copy buffer data into edge adding a 1 pixel padding
  for (i=1; i<MP+1; ++i) {
    for (j=1; j<NP+1; ++j) {
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
    // Do halo swaps
    MPI_Issend(&old[MP][1], 1, column, rank_right, tag_right, cart_comm, &request_right);
    MPI_Issend(&old[1][1],  1, column, rank_left,  tag_left,  cart_comm, &request_left);
    
    MPI_Recv(&old[0][1],    1, column, rank_left,  tag_right, cart_comm, &status);
    MPI_Recv(&old[MP+1][1], 1, column, rank_right, tag_left,  cart_comm, &status);
    
    MPI_Issend(&old[1][NP], 1, row, rank_up,   tag_up,   cart_comm, &request_up);
    MPI_Issend(&old[1][1],  1, row, rank_down, tag_down, cart_comm, &request_down);
    
    MPI_Recv(&old[1][0],    1, row, rank_down, tag_up,   cart_comm, &status);
    MPI_Recv(&old[1][NP+1], 1, row, rank_up,   tag_down, cart_comm, &status);
    
    // Calculate the new value
    for (i=1; i<MP+1; ++i) {
      for (j=1; j<NP+1; ++j) {
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
      
      MPI_Allreduce(&local_delta, &global_delta, 1, MPI_FLOAT, MPI_MAX, cart_comm);
    }
    
    // Update the old value with the new value
    for (i=1; i<MP+1; ++i) {
      for (j=1; j<NP+1; ++j) {
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
  if (rank == root) {
    // Copy local buffer data
    for (i=0; i<MP; ++i) {
      for (j=0; j<NP; ++j) {
        master_buffer[i][N-NP+j] = buffer[i][j];
      }
    }
    
    // Get other processes' buffer data
    int x, y, p;
    for (p=1; p<P; ++p) {
      MPI_Recv(&(buffer[0][0]), MP*NP, MPI_FLOAT, p, tag, cart_comm, &status);
      
      MPI_Cart_coords(cart_comm, p, ndims, coords);
      
      x =  M - (M - coords[1] * MP);
      y =  N - ((coords[0] + 1) * NP);
      
      for (i=0; i<MP; ++i) {
        for (j=0; j<NP; ++j) {
          master_buffer[i+x][j+y] = buffer[i][j];
        }
      }
    }
  } else {
    // In other processes just send the data
    MPI_Ssend(&(buffer[0][0]), MP*NP, MPI_FLOAT, root, tag, cart_comm);
  }
  
  // With master process, write the result image
  if (rank == root) {
    printf("Finished with %d iterations\n", iter);
    
    file_name="result.pgm";
    printf("Writing <%s>\n", file_name);
    pgmwrite(file_name, master_buffer, M, N);
  }
  
  // Finalize the MPI environment
  MPI_Finalize();
  
  return 0;
}
