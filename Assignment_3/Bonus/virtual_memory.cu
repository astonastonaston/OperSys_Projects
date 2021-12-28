#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__device__ void init_invert_page_table(VirtualMemory *vm) {
  // [0,PE-1] valid bits and local count
  // [PE,2*PE-1] page number
  // [3*PE] global count
  // [2*PE, 3*PE-1] PID
  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    vm->invert_page_table[i] = 0x80000000;
    vm->invert_page_table[i + vm->PAGE_ENTRIES] = i; // vpn -> ppn mapping 
    vm->invert_page_table[i + 2*vm->PAGE_ENTRIES] = -1; // process id
  }
  vm->invert_page_table[3*vm->PAGE_ENTRIES] = 0; // timer count
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES, int pid) {
  // init variables
  vm->pid = pid;
  vm->buffer = buffer;
  vm->storage = storage;
  vm->invert_page_table = invert_page_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;

  // init constants
  vm->PAGESIZE = PAGESIZE;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  vm->STORAGE_SIZE = STORAGE_SIZE;
  vm->PAGE_ENTRIES = PAGE_ENTRIES;

  init_invert_page_table(vm);
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
  /* Complate vm_read function to read single element from data buffer */
  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {

    // valid block dealing
    if (!(vm->invert_page_table[i] & 0x80000000)) {
      // page hit
      if ((vm->invert_page_table[i + vm->PAGE_ENTRIES] == (addr/32)) && (vm->invert_page_table[i+2*vm->PAGE_ENTRIES] == threadIdx.x)) {
        // updata count
        vm->invert_page_table[i] &= 0xF0000000;
        vm->invert_page_table[i] += vm->invert_page_table[3*vm->PAGE_ENTRIES];
        vm->invert_page_table[3*vm->PAGE_ENTRIES] += 1;
	return vm->buffer[i*32+addr%32];
      }
      // page miss -> continue
    }

    // invalid block dealing
    else {
      // put disk page in memory
      for (int j = 0; j < vm->PAGESIZE; j++) {
        vm->buffer[i*32+j] = vm->storage[32*(addr/32)+j];
      }
      // update page table
      vm->invert_page_table[i + vm->PAGE_ENTRIES] = addr/32;
      vm->invert_page_table[i] &= 0x00000000;
      vm->invert_page_table[i+2*vm->PAGE_ENTRIES] = threadIdx.x;
      // updata count
      vm->invert_page_table[i] += vm->invert_page_table[3*vm->PAGE_ENTRIES];
      vm->invert_page_table[3*vm->PAGE_ENTRIES] = vm->invert_page_table[3*vm->PAGE_ENTRIES] + 1;
      // update page fault count
      *vm->pagefault_num_ptr = *vm->pagefault_num_ptr + 1;
      return vm->buffer[i*32+addr%32];
    }
  }

  // page fault and page table full
  int least_count = (1<<24), lruind, disk_pn;
  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    if (((vm->invert_page_table[i]) & 0x0FFFFFFF) <= least_count) {
	lruind = i;
    	least_count = vm->invert_page_table[i] & 0x0FFFFFFF;
    }
  }
  disk_pn = vm->invert_page_table[lruind+vm->PAGE_ENTRIES];
  
  // write page back to disk
  // and update memory
  for (int i = 0; i < vm->PAGESIZE; i++) {
    vm->storage[32*disk_pn+i] = vm->buffer[lruind*32+i];
    vm->buffer[lruind*32+i] = vm->storage[32*(addr/32)+i];
  }

  // update current page table and count
  vm->invert_page_table[3*vm->PAGE_ENTRIES] = vm->invert_page_table[3*vm->PAGE_ENTRIES] + 1;
  vm->invert_page_table[lruind] &= 0x00000000;
  vm->invert_page_table[lruind] += vm->invert_page_table[3*vm->PAGE_ENTRIES];
  vm->invert_page_table[lruind+2*vm->PAGE_ENTRIES] = threadIdx.x;
  vm->invert_page_table[lruind + vm->PAGE_ENTRIES] = addr/32;

  // incr pgfault count
  *vm->pagefault_num_ptr = *vm->pagefault_num_ptr + 1;

  return vm->buffer[lruind*32+(addr%32)];
}



__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
  // scanning for page hit cases
  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {

    // valid block dealing
    if (!(vm->invert_page_table[i] & 0x80000000)) {
      // page hit
      if ((vm->invert_page_table[i + vm->PAGE_ENTRIES] == (addr/32)) && (vm->invert_page_table[i+2*vm->PAGE_ENTRIES] == threadIdx.x)) {
        // write to memory buffer
        vm->buffer[i*32+addr%32] = value;

        // updata count
        vm->invert_page_table[i] &= 0xF0000000;
        vm->invert_page_table[i] += vm->invert_page_table[3*vm->PAGE_ENTRIES];
        vm->invert_page_table[3*vm->PAGE_ENTRIES] += 1;
        return ;
      }
      // page miss -> continue
    }

    // invalid block dealing
    else {
      // put disk page in memory
      for (int j = 0; j < vm->PAGESIZE; j++) {
        vm->buffer[i*32+j] = vm->storage[32*(addr/32)+j];
      }
      // write to memory buffer
      vm->buffer[i*32+addr%32] = value;

      // update page table vpn and dirty&invalid bit 
      // and count
      vm->invert_page_table[i + vm->PAGE_ENTRIES] = addr/32;

      vm->invert_page_table[i] &= 0x00000000;
      vm->invert_page_table[i] += vm->invert_page_table[3*vm->PAGE_ENTRIES];
      vm->invert_page_table[i+2*vm->PAGE_ENTRIES] = threadIdx.x;
      vm->invert_page_table[3*vm->PAGE_ENTRIES] = vm->invert_page_table[3*vm->PAGE_ENTRIES] + 1;
      // update page fault count
      *vm->pagefault_num_ptr = *vm->pagefault_num_ptr + 1;

      return ;
    }
  }  

  // page fault and page table full
  int least_count = (1<<24), lruind, disk_pn;
  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    if ((vm->invert_page_table[i] & 0x0FFFFFFF) <= least_count) {
	lruind = i;
    	least_count = vm->invert_page_table[i] & 0x0FFFFFFF;
    }
  }
  disk_pn = vm->invert_page_table[lruind+vm->PAGE_ENTRIES];
  
  // write page back to disk
  // and update memory
  for (int i = 0; i < vm->PAGESIZE; i++) {
    vm->storage[32*disk_pn+i] = vm->buffer[lruind*32+i];
    vm->buffer[lruind*32+i] = vm->storage[32*(addr/32)+i];
  }
  vm->buffer[lruind*32+(addr%32)] = value;
  
  // update current page table and count
  vm->invert_page_table[lruind] &= 0x00000000;
  vm->invert_page_table[lruind] += vm->invert_page_table[3*vm->PAGE_ENTRIES];
  vm->invert_page_table[lruind + vm->PAGE_ENTRIES] = addr/32; 
  vm->invert_page_table[lruind+2*vm->PAGE_ENTRIES] = threadIdx.x;
  vm->invert_page_table[3*vm->PAGE_ENTRIES] = vm->invert_page_table[3*vm->PAGE_ENTRIES] + 1;
  
  // incr pgfault count
  *vm->pagefault_num_ptr = *vm->pagefault_num_ptr + 1;
  
  return ;
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
  for (int i=offset; i<input_size; i++) {
    results[i] = vm_read(vm, i);
  }
}

