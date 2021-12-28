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
#define RM_RF 3
#define PWD 4
#define CD 5
#define CD_P 6
#define MKDIR 7

// in total 44 bytes
typedef struct {
  char name[20]; // 20-byte name
  u32 modify_cnt; // 4-byte modify cnt 
  u32 create_cnt; // 4-byte create cnt 
  uint16_t size; // 2-byte size (lower 11-bit size, highest 1-bit is_directory)
  uint16_t file_index; // 2-byte block index (lower 15-bit file block index, lowest 1-bit valid)
  u32* next_fcb_ptr; // 4-byte next fcb ptr
  u32* prev_fcb_ptr; // 4-byte prev fcb ptr
  u32* subdir_ptr; // 4-byte subdir ptr
} __attribute__((__packed__)) FCBlock;

struct FileSystem {
	uchar *volume;
	FCBlock *FCB_curr_ptr;
	uchar *file_curr_ptr; // pointing to the current empty file block
	uchar *bitmap_curr_ptr; // current bitmap location (the nearest location where *bitmap_curr_ptr != 0xFF)
	int bitmap_curr_offset; // bit offset at bitmap_curr_ptr (where the bit = 0), value ∈ [0,7]
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


__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
	int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
	int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE,
	int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS);

__device__ int dev_abs(int val);
__device__ u32 fs_open(FileSystem *fs, char *s, int op);
__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp);
__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp);
__device__ void fs_gsys(FileSystem *fs, int op);
__device__ void fs_gsys(FileSystem *fs, int op, char *s);
__device__ void dev_memset(char* start, uint8_t value, int size);
__device__ bool dev_strcmp(char* a, char* b);
__device__ void kuaipai(FCBlock* ptrs[], int op, int size); 
__device__ void dev_strcpy(char* dest, char* src);
__device__ int dev_strlen(char* str);

#endif
