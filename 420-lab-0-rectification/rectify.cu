#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"
#include <stdlib.h>
#include <stdio.h>

__global__ void rectify() {
  //stuff
  printf("hello!");
}

int main(int argc, char** argv) {
  //stuff
  rectify<<<1, 1>>>();
  return 0;
}
