
#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __managed__ u32 gtime = 1;

// sort the FCBs according to modified time/size using quicksort (ptrs[i]->size/ptrs[i]->modify_cnt)
__device__ void kuaipai(FCBlock* ptrs[], int op, int size) 
{
  if (op == 1) { // LS_S
    if (size == 0 || size == 1) return ;
    int j=-1, pivot=size-1;
    FCBlock* k;
    for (int i = 0; i < size; i++) {
      // first create first cnt
      if ((ptrs[i]->size < ptrs[pivot]->size) || ((ptrs[i]->size == ptrs[pivot]->size)&&(ptrs[i]->create_cnt >= ptrs[pivot]->create_cnt))) {        
	k = ptrs[i];
        ptrs[i] = ptrs[j+1];
        ptrs[j+1] = k;
        j++;
      }
    }
    // pivot arrived at suitable location
    kuaipai(ptrs, 1, j);
    kuaipai(ptrs+j+1, 1, size-j-1);
  }
  else if (op == 0) { // LS_D
    if (size == 0 || size == 1) return ;
    int j=-1, pivot=size-1;
    FCBlock* k;
    for (int i = 0; i < size; i++) {
      if (ptrs[i]->modify_cnt <= ptrs[pivot]->modify_cnt) {
        k = ptrs[i];
        ptrs[i] = ptrs[j+1];
        ptrs[j+1] = k;
        j++;
      }
    }
    // pivot arrived at suitable location
    kuaipai(ptrs, 0, j);
    kuaipai(ptrs+j+1, 0, size-j-1);
  }
}

// absolute value
__device__ int dev_abs(int val) {
  if (val >= 0) return val;
  else return -val;
}

// 0->match 1->unmatch
__device__ bool dev_strcmp(char* a, char* b) {
  int i;
  for (i = 0; (a[i] != '\0' && b[i] != '\0'); i++) {
    if (a[i] != b[i]) return 1;
  }
  if (a[i] == b[i]) return 0;
  else return 1;
}

// set(init) memory
__device__ void dev_memset(uchar* start, uint8_t value, int size) {
  for (int i = 0; i < size; i++) start[i] = value;
}

// string copy
__device__ void dev_strcpy(char* dest, char* src) {
  int i;
  for (i = 0; src[i] != '\0'; i++) dest[i] = src[i];
  dest[i] = src[i];
}

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;
  // 4KB bitmap free space manager
  // 32KB FCB
  // 1024KB file content

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;
  fs->bitmap_curr_ptr = fs->volume;
  fs->bitmap_curr_offset = 7; // index of the first 0 bit (from right) in *bitmap_curr_ptr 
  fs->FCB_curr_ptr = (FCBlock*)(fs->volume + SUPERBLOCK_SIZE);
  fs->file_curr_ptr = fs->volume + fs->FILE_BASE_ADDRESS; // the most recent empty file block location

  // init FCBlocks
  FCBlock blk;
  blk.size=0;
	blk.file_index=0;
	blk.modify_cnt=0;
	blk.create_cnt=0;
  for (int i = 0; i <= 19; i++) blk.name[i]=0;
  for (FCBlock* i = fs->FCB_curr_ptr; i <= (fs->FCB_curr_ptr+1023); i++) *i = blk;
  for (int i = 0; i <= VOLUME_SIZE-1; i++) fs->volume[i] = 0;
}

__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
  // location ptr init
  if (s[0] == '\0') {
    printf("Open Error: at least give me a non-empty file name?\n");
    return ;
  }
  FCBlock * fcbPtr = (FCBlock*)(fs->volume + fs->SUPERBLOCK_SIZE), * iniFcbPtr = fcbPtr; // [4k - 36k-1]

  // existing, just return ptr
  while (true) {
    // valid block judgement
    if (fcbPtr->file_index & (0x01) == 1) {
      // file name equal judgement
      if (!(dev_strcmp(s, fcbPtr->name))) {
        // if existing, emptify and return fcb volume index directly
    	  if (op==G_WRITE) fs_write(fs, (uchar*)s, 0, (((4*1024 + (fcbPtr-iniFcbPtr)*32) & 0x7FFFFFFF) + (G_WRITE<<31)));
        return (((4*1024 + (fcbPtr-iniFcbPtr)*32) & 0x7FFFFFFF) + (op<<31));
      }
      fcbPtr++;
    }
    else break; // terminating, fcbPtr pointing to the nearest invalid block
  }

  // not existing
  // FCB update
  dev_strcpy((char*)fcbPtr->name, s);
  // update size
  fcbPtr->size = 0;
  // update valid bit
  fcbPtr->file_index = 1;
  // update block index
  fcbPtr->file_index = fcbPtr->file_index + ((fs->file_curr_ptr - fs->volume - fs->FILE_BASE_ADDRESS) << 1);
  // update modify/create cnt
  fcbPtr->modify_cnt = gtime++;
  fcbPtr->create_cnt = fcbPtr->modify_cnt;

  // fs current ptr updates (incr 32 Bytes)
  fs->FCB_curr_ptr = fs->FCB_curr_ptr + 1;
  fs->file_curr_ptr = fs->file_curr_ptr + 32;

  // bitmap allocation
  // bit allocation
  *(fs->bitmap_curr_ptr) = *(fs->bitmap_curr_ptr) + (1 << fs->bitmap_curr_offset);
  fs->bitmap_curr_offset = fs->bitmap_curr_offset - 1;
      
  // update current bitmap ptr
  if (fs->bitmap_curr_offset < 0) {
    fs->bitmap_curr_offset = 7;
    fs->bitmap_curr_ptr = fs->bitmap_curr_ptr + 1;
  }

  // G_WRITE emptify and return fp
  if (op==G_WRITE) fs_write(fs, (uchar*)s, 0, ((4*1024 + (fcbPtr-iniFcbPtr)*32) + (G_WRITE<<31)));
  return ((4*1024 + (fcbPtr-iniFcbPtr)*32) + (op<<31));
}


__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp) // fp -> FCB's volume index!
{
  if (fp==0) {
    printf("Oops, it seems that the file has some open error, so it cannot be read\n");
    return ;
  }

  if ((fp&0x80000000)==0x80000000) {
    printf("Read Error: write-only file does not support reading\n");
    return;
  }
  FCBlock * fcbTarget = ((fp-1024*4)/32 + (FCBlock*)(fs->volume + fs->SUPERBLOCK_SIZE));

  uchar* file_ptr = (fs->volume + fs->FILE_BASE_ADDRESS)+32*((fcbTarget->file_index & 0xFFFE) >> 1);

  for (int i = 0; i < size; i++) output[i] = (file_ptr)[i];
}


__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp) // fp -> FCB's volume index!
{
  if (fp==0) {
    printf("Oops, it seems that the file has some open error, so it cannot be written\n");
    return ;
  }
  if ((fp&0x80000000)==0) {
    printf("Write Error: writing to a read-only file\n");
    return;
  }
  // Translate to target address
  if (size<0) {printf("Write Error: negative size\n"); return;}
  
  FCBlock * fcbTarget = (((fp&0x7FFFFFFF)-1024*4)/32 + (FCBlock*)(fs->volume + fs->SUPERBLOCK_SIZE));
  uchar* file_ptr = (fs->volume + fs->FILE_BASE_ADDRESS)+32*((fcbTarget->file_index & 0xFFFE) >> 1);
   
  // size judgement
  int original_size = fcbTarget->size;

  // calculate block size difference
  int original_block_num = (original_size-1)/32 + 1, new_block_num = dev_abs(size-1)/32 + 1, block_size_diff = new_block_num - original_block_num; 
  
  // bitmap reallocation
  if (block_size_diff > 0) { // block size needs extending
    // reallocate in bitmap
    // enough bits for reallocation at the current bitmap block
    if (fs->bitmap_curr_offset+1 >= block_size_diff) { 
      // bit allocation
      for (int i = 0; i < block_size_diff; i++) {
        *(fs->bitmap_curr_ptr) = *(fs->bitmap_curr_ptr) + (1 << fs->bitmap_curr_offset);
        fs->bitmap_curr_offset = fs->bitmap_curr_offset - 1;
      }
      
      // update current bitmap ptr
      if (fs->bitmap_curr_offset < 0) {
        fs->bitmap_curr_offset = 7;
        fs->bitmap_curr_ptr = fs->bitmap_curr_ptr + 1;
      }
    }

    // not enough bits at current bitmap block
    // edge case: scaling at boundry
    else { 
      // current bitmap block filled
      *(fs->bitmap_curr_ptr) = 0xFF;
     
      // next bitmap block allocated
      fs->bitmap_curr_ptr = fs->bitmap_curr_ptr+1;
      int old_offset = fs->bitmap_curr_offset;
      fs->bitmap_curr_offset = 7;
      
      for (int i = 0; i < block_size_diff-(old_offset+1); i++) {
        *(fs->bitmap_curr_ptr) = *(fs->bitmap_curr_ptr) + (1 << (7-(i%8)));
        fs->bitmap_curr_offset = fs->bitmap_curr_offset - 1;
        if (i%8==7) {
          fs->bitmap_curr_ptr = fs->bitmap_curr_ptr+1;
          fs->bitmap_curr_offset = 7;
        }
      }

      // translate back to bit_offset and bit_curr_ptr
    }

    // compact in file space
    uchar* compactStt = file_ptr + 32*original_block_num, * compactEnd = fs->file_curr_ptr;
    for (int i = 0; i < compactEnd-compactStt; i++) *(compactStt + block_size_diff*32 + i) = compactStt[i]; 
    for (int i = 0; i < block_size_diff*32; i++) compactStt[i] = 0;
  }

  else if (block_size_diff < 0) { // block size needs shrinking
    // reallocate in bitmap
    // enough bits for shrink
    block_size_diff = block_size_diff*(-1); 
    if (7-(fs->bitmap_curr_offset) >= block_size_diff) { 
      // bit allocation
      for (int i = 0; i < block_size_diff; i++) {
        fs->bitmap_curr_offset = fs->bitmap_curr_offset + 1;
        *(fs->bitmap_curr_ptr) = *(fs->bitmap_curr_ptr) - (1 << fs->bitmap_curr_offset);
      }
    }

    // not enough bits at current bitmap block
    else { 
      // current bitmap block filled
      *(fs->bitmap_curr_ptr) = 0x00;
     
      // next bitmap block allocated
      int old_offset = fs->bitmap_curr_offset;
      fs->bitmap_curr_offset = 7;
      
      for (int i = 0; i < block_size_diff-(7-old_offset); i++) {
        if (i%8==0) {
          fs->bitmap_curr_ptr=fs->bitmap_curr_ptr-1;
      	  fs->bitmap_curr_offset = -1;
	      }

        *(fs->bitmap_curr_ptr) = *(fs->bitmap_curr_ptr) - (1 << (i%8));
        fs->bitmap_curr_offset = fs->bitmap_curr_offset + 1;
      }
    }

    // compact in file space
    uchar* compactStt = file_ptr + 32*original_block_num, * compactEnd = fs->file_curr_ptr;
    for (int i = 0; i < compactEnd-compactStt; i++) *(compactStt - block_size_diff*32 + i) = compactStt[i]; 
    for (int i = 0; i < block_size_diff*32; i++) (fs->file_curr_ptr - block_size_diff*32)[i] = 0;

    block_size_diff = block_size_diff*(-1); 
  }

  // write contents inside 
  for (int i = 0; i < size; i++) *((file_ptr) + i) = input[i]; 
  
  // update FCB of written file (oh but, Hala Madrid!)
  fcbTarget->size = size;
  fcbTarget->modify_cnt = gtime++;

  uchar* ite_ptr; 

  // update FCB of latter files (file pointer backshift/foreshift)
  for (int i = 1; (fcbTarget+i)->file_index & 1 != 0; i++) {
    (fcbTarget+i)->file_index = ((((fcbTarget+i)->file_index >> 1) + block_size_diff) << 1) + 1;
  }
  fs->file_curr_ptr = fs->file_curr_ptr + block_size_diff*32;

  uchar* filstt = fs->volume + fs->FILE_BASE_ADDRESS;
  return fp;
}
__device__ void fs_gsys(FileSystem *fs, int op)
{
  FCBlock* FCB_stt_ptr = (FCBlock*)(fs->volume + fs->SUPERBLOCK_SIZE);
  int ptrlen = (fs->FCB_curr_ptr) - FCB_stt_ptr;
  FCBlock** blockPtrs = new FCBlock*[ptrlen];
  
  // init ptr array
  for (FCBlock* i = fs->FCB_curr_ptr-1; i >= FCB_stt_ptr; i--) {
    blockPtrs[i - FCB_stt_ptr] = i;
  }

  // sort the ptr array
  kuaipai(blockPtrs, op, fs->FCB_curr_ptr - FCB_stt_ptr);

  if (op == 1) { // LS_S (order by size)
    printf("===sort by file size===\n");
    // descending order
    for (FCBlock* i = fs->FCB_curr_ptr-1; i >= FCB_stt_ptr; i--) {
      printf("%s %d\n", blockPtrs[i - FCB_stt_ptr]->name, blockPtrs[i - FCB_stt_ptr]->size);
    }
  }

  else if (op == 0) { // LS_D (order by mod_time)
    printf("===sort by modified time===\n");
    // descending order
    for (FCBlock* i = fs->FCB_curr_ptr-1; i >= FCB_stt_ptr; i--) {
      printf("%s\n", blockPtrs[i - FCB_stt_ptr]->name);
    }
  }
  delete [] blockPtrs;
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
  if (op == 2) { // RM
    FCBlock* rm_ptr=0, * FCB_stt_ptr = (FCBlock*)(fs->volume + fs->SUPERBLOCK_SIZE);
    // find the rm ptr
    for (FCBlock* i = fs->FCB_curr_ptr-1; i >= FCB_stt_ptr; i--) {
      if (dev_strcmp(i->name, s) == 0) {
        rm_ptr = i;
      }
    }

    if (rm_ptr==0) {
	    printf("Remove Error: No such file or directory\n");
      return;
    }
    uchar* file_ptr = (fs->volume + fs->FILE_BASE_ADDRESS)+32*((rm_ptr->file_index & 0xFFFE) >> 1);

    // reallocate free space
    int deallo_block_num=(dev_abs(rm_ptr->size-1))/32 + 1;
    if ((7-fs->bitmap_curr_offset) >= deallo_block_num) { // enough bits for deallocation
      for (int i = 0; i < deallo_block_num; i++) {
        fs->bitmap_curr_offset = fs->bitmap_curr_offset + 1;
        *(fs->bitmap_curr_ptr) = *(fs->bitmap_curr_ptr) - (1 << fs->bitmap_curr_offset);
      }
    }
    else { // not enough bits for deallocation
      // current bitmap block filled
      *(fs->bitmap_curr_ptr) = 0x00;
     
      // next bitmap block allocated
      int old_offset = fs->bitmap_curr_offset;
      fs->bitmap_curr_offset = 7;
      
      for (int i = 0; i < deallo_block_num-(7-old_offset); i++) {
        if (i%8==0) {
          fs->bitmap_curr_ptr=fs->bitmap_curr_ptr-1;
      	  fs->bitmap_curr_offset = -1;
	      }

        *(fs->bitmap_curr_ptr) = *(fs->bitmap_curr_ptr) - (1 << (i%8));
        fs->bitmap_curr_offset = fs->bitmap_curr_offset + 1;
      }
    }
    
    // file space compaction
    for (int i = 0; i < (fs->file_curr_ptr - file_ptr - 32 * deallo_block_num + 1); i++) (file_ptr)[i] = (file_ptr + 32 * deallo_block_num)[i]; 
    
    // nullify the last FCB block
    fs->file_curr_ptr = fs->file_curr_ptr - 32 * deallo_block_num;
    for (int i = 0; i < 32 * deallo_block_num; i++) (file_ptr)[i] = 0;

    for (FCBlock* i = rm_ptr+1; i <= fs->FCB_curr_ptr-1; i++) *(i-1) = *i;
    fs->FCB_curr_ptr = fs->FCB_curr_ptr - 1;
    (fs->FCB_curr_ptr)->size=0;
    (fs->FCB_curr_ptr)->file_index=0;
    (fs->FCB_curr_ptr)->create_cnt=0;
    (fs->FCB_curr_ptr)->modify_cnt=0;
    for (int i = 0; i <= 19; i++) (fs->FCB_curr_ptr)->name[i]=0;

    // fcb file ptr updates
    for (FCBlock* i = fs->FCB_curr_ptr-1; i >= rm_ptr; i--) {
      i->file_index = (((i->file_index>>1) - deallo_block_num)<<1) + 1;
    }

    uchar* filstt = fs->volume + fs->FILE_BASE_ADDRESS;
  }
}
