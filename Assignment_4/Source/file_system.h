#ifndef VIRTUAL_MEMORY_H
#define VIRTUAL_MEMORY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>

typedef unsigned char uchar;
typedef uint32_t u32;


#define G_WRITE 1
#define G_READ 0
#define LS_D 0
#define LS_S 1
#define RM 2

// in total 32 bytes
typedef struct {
  char name[20]; // 20-byte name
  u32 modify_cnt; // 4-byte modify cnt 
  u32 create_cnt; // 4-byte create cnt 
  uint16_t size; // 2-byte size (11-bit size)
  uint16_t file_index; // 2-byte block index (lower 15-bit file block index, highest 1-bit valid)
} __attribute__((__packed__)) FCBlock;

struct FileSystem {
	uchar *volume;
	FCBlock *FCB_curr_ptr; // pointing to the closest empty FCB 
	uchar *file_curr_ptr; // pointing to the closest empty file block
	uchar *bitmap_curr_ptr; // current bitmap location (the nearest location where *bitmap_curr_ptr != 0xFF)
	int bitmap_curr_offset; // bit offset (where the offseted bit = 0) at bitmap_curr_ptr, value ∈ [0,7]
	int SUPERBLOCK_SIZE;
	int FCB_SIZE;
	int FCB_ENTRIES;
	int STORAGE_SIZE;
	int STORAGE_BLOCK_SIZE;
	int MAX_FILENAME_SIZE;
	int MAX_FILE_NUM;
	int MAX_FILE_SIZE;
	int FILE_BASE_ADDRESS;
};

__device__ int dev_abs(int val);
__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
	int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
	int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE,
	int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS);

__device__ u32 fs_open(FileSystem *fs, char *s, int op);
__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp);
__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp);
__device__ void fs_gsys(FileSystem *fs, int op);
__device__ void fs_gsys(FileSystem *fs, int op, char *s);
__device__ void dev_memset(char* start, uint8_t value, int size);
__device__ bool dev_strcmp(char* a, char* b);
__device__ void kuaipai(FCBlock* ptrs[], int op, int size); 
__device__ void dev_strcpy(char* dest, char* src);

#endif
