#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "lodepng.h"

__global__ void rectify(unsigned char image[], int len) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index > len) return;
  if (image[index] < 127)
    image[index] = 127;
}

int main(int argc, char** argv) {
  char* input_name = argv[1];
  char* output_name = argv[2];
  int no_threads = atoi(argv[3]);

  unsigned char *image, *d_image;
  unsigned int width, height;

  unsigned int error = lodepng_decode32_file((unsigned char**)&image, &width, &height, input_name);
  if (error) {
    printf("error %u: %s\n", error, lodepng_error_text(error));
    exit(-1);
  }

  cudaMalloc((void**)&d_image, width * height * 4 * sizeof(unsigned char));
  cudaMemcpy((void*)d_image, (void*)image, width * height * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);

  //start timer
  float memsettime;
  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  rectify<<<(width * height * 4 + no_threads - 1) / no_threads, no_threads>>>(d_image, width * height * 4);

  //stop timer
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&memsettime, start, stop);
  printf("Kernel execution time: %f\n", memsettime);
  cudaEventDestroy(start); cudaEventDestroy(stop);

  cudaMemcpy((void*)image, (void*)d_image, width * height * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  lodepng_encode32_file(output_name, image, width, height);

  free(image);
  cudaFree(d_image);

  return 0;
}
