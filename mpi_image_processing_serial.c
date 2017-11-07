#include <stdio.h>
#include <stdlib.h>

#include "pgmio.h"

#define NUM_ITER 100

int main(int argc, char *argv[]) {
  char *file_name;
  int M, N;
  int i, j, iter;
  
  file_name = "edge192x128.pgm";
  pgmsize(file_name, &M, &N);
  
  float buffer[M][N];
  float old[M+2][N+2];
  float new[M+2][N+2];
  float edge[M+2][N+2];
  
  // Read the image data into buffer
  printf("Reading <%s>\n", file_name);
  pgmread(file_name, buffer, M, N);
  
  // Copy buffer data into edge adding a 1 pixel padding
  for (i=1; i<=M; ++i) {
    for (j=1; j<=N; ++j) {
      edge[i][j] = buffer[i-1][j-1];
    }
  }
  
  // Initialize old values with 255.0 (white)
  for (i=0; i<M+2; ++i) {
    for (j=0; j<N+2; ++j) {
      old[i][j] = 255.0; 
    }
  }
  
  // Main loop
  for (iter=0; iter<NUM_ITER; ++iter) {
    // Calculate the new value
    for (i=1; i<=M; ++i) {
      for (j=1; j<=N; ++j) {
        new[i][j] = 0.25 * (old[i-1][j] + old[i+1][j] + 
                            old[i][j-1] + old[i][j+1] - edge[i][j]); 
      }
    }
    
    // Update the old value with the new value
    for (i=1; i<=M; ++i) {
      for (j=1; j<=N; ++j) {
        old[i][j] = new[i][j]; 
      }
    }
  }
  
  // Copy data back to buffer
  for (i=0; i<M; ++i) {
    for (j=0; j<N; ++j) {
      buffer[i][j] = old[i+1][j+1];
    }
  }
  
  // Write the result image
  file_name="result192x128.pgm";
  printf("Writing <%s>\n", file_name);
  pgmwrite(file_name, buffer, M, N);
  
  return 0;
}