#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
                             int input_size) {
  int value;

  for (int i = 0; i < input_size; i++)
    vm_write(vm, i, input[i]);

  for (int i = input_size - 1; i >= input_size - 32769; i--) {
    value = vm_read(vm, i);
  }

  // Load physical memory (RAM) in shared memory to results buffer in global memory
  vm_snapshot(vm, results, 0, input_size);
}
