#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"
#include <stdlib.h>


__global__ void pool(unsigned char output_image[], const unsigned char image[], unsigned int width_out, unsigned int height_out) {
  // perspective: image is an array of pixel (x), with 4 channels (z) per pixel
  // A thread is given to each output channel per pixel.
  int pixel_out = blockDim.x * blockIdx.x + threadIdx.x;
  int index_out = pixel_out * blockDim.z + threadIdx.z;  // index = pixel + channel
  int y_out = pixel_out / width_out;
  int x_out = pixel_out - y_out * width_out;
  // pixel index of the input image. Each output pixel is mapped to the the first 2x2 pixel block of the input image.
  int pixel_in = (y_out * width_out * 4 + x_out * 2) * 4;
  //printf("Pixel_in = %d  y_out = %d  x_out = %d\n", pixel_in, y_out, x_out);
  if (index_out >= width_out * height_out * 4) return;

  // look at each pixel value in the 2x2 pixel block, choose the largest value.
  int max = 0;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      int val = image[pixel_in + 4 * i + 8 * width_out * j + threadIdx.z];
      if (val > max)
        max = val;
    }
  }

  output_image[index_out] = max;
}

int main(int argc, char** argv) {
  char* input_name = argv[1];
  char* output_name = argv[2];
  int no_threads = atoi(argv[3]);

  unsigned char *old_image, *d_old_image, *new_image;
  unsigned int width, height;

  unsigned int error = lodepng_decode32_file((unsigned char**)&old_image, &width, &height, input_name);
  if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

  cudaMalloc((void**)&d_old_image, width * height * 4 * sizeof(unsigned char));
  cudaMemcpy((void*)d_old_image, (void*)old_image, width * height * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMallocManaged((void**)&new_image, width * height * sizeof(unsigned char)); // width/2 * height/2 * 4channels = width * height
  
  pool<<<(width * height + (no_threads * 4) - 1) / (no_threads * 4), dim3(no_threads, 1, 4)>>>(new_image, d_old_image, width / 2, height / 2);

  cudaDeviceSynchronize();

  lodepng_encode32_file(output_name, new_image, width / 2, height / 2);

  free(old_image);
  cudaFree(d_old_image);
  cudaFree(new_image);

  return 0;
}