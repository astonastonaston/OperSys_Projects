
#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#define FCBLOCKSIZE 44
__device__ __managed__ u32 gtime = 1;
__device__ __managed__ FCBlock* curr_dir = 0;
__device__ __managed__ FCBlock* par_dir = 0;
__device__ __managed__ FCBlock* pwd[4] = {0, 0, 0, 0}; // storing ptrs to sub-dirs

// absolute value
__device__ int dev_abs(int val) {
  if (val >= 0) return val;
  else return -val;
}

// sort the FCBs according to modified time/size (ptrs[i]->size/ptrs[i]->modify_cnt) using quicksort
__device__ void kuaipai(FCBlock* ptrs[], int op, int size) 
{
  size = size & 0x7FFF;
  if (op == 1) { // LS_S
    if (size == 0 || size == 1) return ;
    int j=-1, pivot=size-1;
    FCBlock* k;
    for (int i = 0; i < size; i++) {
      if (((ptrs[i]->size & 0x7FFF) < (ptrs[pivot]->size & 0x7FFF)) || (((ptrs[i]->size & 0x7FFF) == (ptrs[pivot]->size& 0x7FFF))&&(ptrs[i]->create_cnt >= ptrs[pivot]->create_cnt))) {        
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

// 0->match 1->unmatch
__device__ bool dev_strcmp(char* a, char* b) {
  int i;
  for (i = 0; (a[i] != '\0' && b[i] != '\0'); i++) {
    if (a[i] != b[i]) return 1;
  }
  if (a[i] == b[i]) return 0;
  else return 1;
}

// memory set (init)
__device__ void dev_memset(uchar* start, uint8_t value, int size) {
  for (int i = 0; i < size; i++) start[i] = value;
}

// calculate string length
__device__ int dev_strlen(char* str) {
  int i=0;
  for (i; str[i]!='\0'; i++) continue;
  return i;
}

// remove all sub-dirs and files in dfs. Input is the directory
__device__ void dfs_rm(FileSystem *fs, FCBlock* ptr) {
  // rm a file
  FCBlock* tt;
  if ((ptr->size & 0x8000) == 0) {
	  fs_gsys(fs, RM, ptr->name);
  }

  // rm empty dir
  else if (((ptr->size & 0x8000) == 0x8000) && (ptr->subdir_ptr == 0)) {
    // logical fcb rm
    FCBlock* k=(FCBlock *)(ptr->prev_fcb_ptr);

    if ((FCBlock*)(ptr->prev_fcb_ptr) != curr_dir) k->next_fcb_ptr = ptr->next_fcb_ptr;
    else k->subdir_ptr = ptr->next_fcb_ptr;
    
    if (ptr->next_fcb_ptr != 0) ((FCBlock *)(ptr->next_fcb_ptr))->prev_fcb_ptr = (u32*)k;

    // FCB space compaction
    for (FCBlock* i = ptr+1; i <= fs->FCB_curr_ptr-1; i++) { 
      // ending of sub-dir consideration
      if (i->next_fcb_ptr != 0)  ((FCBlock*)(i->next_fcb_ptr))->prev_fcb_ptr = (u32*)((FCBlock*)(((FCBlock*)(i->next_fcb_ptr))->prev_fcb_ptr) - 1);

      // starting sub-dir consideration
      if ( !dev_strcmp( ((FCBlock*)(i->prev_fcb_ptr))->name, curr_dir->name ) ) {
        ((FCBlock*)(i->prev_fcb_ptr))->subdir_ptr = (u32*)((FCBlock*)(((FCBlock*)(i->prev_fcb_ptr))->subdir_ptr) - 1);
      }

      else {
        u32* prvnxt = ((FCBlock*)(i->prev_fcb_ptr))->next_fcb_ptr;
        ((FCBlock*)(i->prev_fcb_ptr))->next_fcb_ptr = (u32*)(((FCBlock*)prvnxt) - 1); 
      }
      
      *(i-1) = *i;
    }

    // nullify curr FCB
    fs->FCB_curr_ptr = fs->FCB_curr_ptr - 1;
    (fs->FCB_curr_ptr)->size=0;
    (fs->FCB_curr_ptr)->file_index=0;
    (fs->FCB_curr_ptr)->create_cnt=0;
    (fs->FCB_curr_ptr)->modify_cnt=0;
    (fs->FCB_curr_ptr)->next_fcb_ptr=0;
    (fs->FCB_curr_ptr)->prev_fcb_ptr=0;
    (fs->FCB_curr_ptr)->subdir_ptr=0;
    for (int i = 0; i <= 19; i++) (fs->FCB_curr_ptr)->name[i]=0;
  }
  // rm all files inside a dir
  else {
    FCBlock* i= (FCBlock*)(ptr->subdir_ptr), *j;
    fs_gsys(fs, CD, ptr->name);

    for (i; i->next_fcb_ptr != 0; i = (FCBlock*)(i->next_fcb_ptr)) continue;
    
    // rm all files inside a dir
    while ((FCBlock*)(i->prev_fcb_ptr) != curr_dir) {
      i = (FCBlock*)(i->prev_fcb_ptr);
      tt = (FCBlock*)(curr_dir->subdir_ptr);
      dfs_rm(fs, (FCBlock*)(i->next_fcb_ptr)); 
    }
    dfs_rm(fs, i); 

    // rm empty dir
    fs_gsys(fs, CD_P);
    dfs_rm(fs, ptr);  
  }
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
	blk.file_index=0; // for sub-dir, it isn't used
	blk.modify_cnt=0;
	blk.create_cnt=0;
	blk.next_fcb_ptr=0;
	blk.prev_fcb_ptr=0;
	blk.subdir_ptr=0;
  for (int i = 0; i <= 19; i++) blk.name[i]=0;
  for (FCBlock* i = fs->FCB_curr_ptr; i <= (fs->FCB_curr_ptr+1023); i++) *i = blk;
  for (int i = 0; i <= VOLUME_SIZE-1; i++) fs->volume[i] = 0;

  // init root dir
  dev_strcpy(blk.name, "root");
  blk.size = (1<<15); // set directory bit
  blk.file_index = 1; // set valid bit
	blk.create_cnt = gtime++;
	blk.modify_cnt = blk.create_cnt;
  *(fs->FCB_curr_ptr) = blk;
  fs->FCB_curr_ptr = fs->FCB_curr_ptr + 1;

  // init curr dir
  curr_dir = fs->FCB_curr_ptr-1;
  pwd[0] = curr_dir;
}



// fcb design: 32B: 20B(name) + 4B(modify_count) + 2B(size, top bit indicating valid) + 4B(ptrToFileSpaceLocation)
__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
  // location ptr init
  if (s[0] == '\0') {
    printf("Open Error: at least give me a non-empty file name?\n");
    return ;
  }
  FCBlock * fcbPtr = (FCBlock*)(fs->volume + fs->SUPERBLOCK_SIZE), * iniFcbPtr = fcbPtr, * trace_currdir_ptr; // [4k - 44k-1]

  // existing, just return ptr
  // find empty space
  while (true) {
    // valid block judgement
    if (fcbPtr->file_index & (0x01) == 1) {
      // file name equal judgement
      if (!(dev_strcmp(s, fcbPtr->name))) {
        // if existing and under the same dir, return fcb volume index directly
        trace_currdir_ptr = fcbPtr;

        // trace back to directory of the file
        while (trace_currdir_ptr->prev_fcb_ptr != 0) trace_currdir_ptr = (FCBlock*)(trace_currdir_ptr->prev_fcb_ptr);

        // if under the same dir, file found
        if (!(dev_strcmp(trace_currdir_ptr->name, curr_dir->name))) {
		if (op==G_WRITE) fs_write(fs, (uchar*)s, 0, (((4*1024 + (fcbPtr-iniFcbPtr)*44) & 0x7FFFFFFF) + (G_WRITE<<31)));
		return ((4*1024 + (fcbPtr-iniFcbPtr)*FCBLOCKSIZE) + (op<<31));
	  }
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
  fcbPtr->next_fcb_ptr = 0;
  fcbPtr->subdir_ptr = 0;

  // directory update
  FCBlock* i = (FCBlock*)(curr_dir->subdir_ptr);
  if (i!=0) {
    for (i; i->next_fcb_ptr != 0; i = (FCBlock*)(i->next_fcb_ptr)) continue;
  }

  // back ptr
  if (i!=0) fcbPtr->prev_fcb_ptr = (u32*)i;
  else fcbPtr->prev_fcb_ptr = (u32*)curr_dir;

  // front ptr
  if (i==0) curr_dir->subdir_ptr = (u32*)fcbPtr;
  else i->next_fcb_ptr = (u32*)fcbPtr;

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

  // update dir size
  curr_dir->size = curr_dir->size + dev_strlen(s);
  curr_dir->modify_cnt = gtime++;

  if (op==G_WRITE) fs_write(fs, (uchar*)s, 0, ((4*1024 + (fcbPtr-iniFcbPtr)*44) + (G_WRITE<<31)));
  return ((4*1024 + (fcbPtr-iniFcbPtr)*FCBLOCKSIZE) + (op<<31));
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
  FCBlock * fcbTarget = ((fp-1024*4)/FCBLOCKSIZE + (FCBlock*)(fs->volume + fs->SUPERBLOCK_SIZE));

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
  if (size<0) {
    printf("Write Error: negative size\n"); 
    return;
  }
  
  // Translate to target address
  size =  size & 0x7FFF;
  FCBlock * fcbTarget = (((fp&0x7FFFFFFF)-1024*4)/FCBLOCKSIZE + (FCBlock*)(fs->volume + fs->SUPERBLOCK_SIZE));
  uchar* file_ptr = (fs->volume + fs->FILE_BASE_ADDRESS)+32*((fcbTarget->file_index & 0xFFFE) >> 1);
   
  // size judgement
  int original_size = fcbTarget->size;

  // calculate block size difference
  int original_block_num = (original_size-1)/32 + 1, new_block_num = dev_abs(size-1)/32 + 1, block_size_diff = new_block_num - original_block_num; 
  
  // bitmap reallocation
  if (block_size_diff > 0) { // block size needs extending
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

  for (int i = 0; i < size; i++) *((file_ptr) + i) = input[i]; 
  
  // update FCB of written file (oh but, Hala Madrid!)
  fcbTarget->size = size;
  fcbTarget->modify_cnt = gtime++;

  // update FCB of latter files (file pointer backshift/foreshift)
  for (int i = 1; (fcbTarget+i)->file_index & 1 != 0; i++) {
    (fcbTarget+i)->file_index = ((((fcbTarget+i)->file_index >> 1) + block_size_diff) << 1) + 1;
  }
  fs->file_curr_ptr = fs->file_curr_ptr + block_size_diff*32;

  curr_dir->modify_cnt = gtime++;
  return fp;
}

__device__ void fs_gsys(FileSystem *fs, int op)
{
  if (op == CD_P) {
    // change pwd
    int i=0;
    for (i; i<=3; i++) {
      if (pwd[i] == curr_dir) break;
    }
    pwd[i]=0;

    // change curr dir and parent dir
    curr_dir=pwd[i-1];

    if (i-2>=0) par_dir=pwd[i-2];
    else par_dir=0;
  }

  else if (op == PWD) {
    int i=1;
    while (pwd[i]!=0 && pwd[i]!='\0') {
      printf("/%s", pwd[i]->name);
      i++;
    }
    printf("\n");
  }
  
  else if (op == LS_D || op == LS_S) {
    int len=0;
    FCBlock* i=(FCBlock*)(curr_dir->subdir_ptr), *k = i;

    if (i!=0) {
      len++; // in case directory has no file
      for (i; i->next_fcb_ptr!=0; i=(FCBlock*)(i->next_fcb_ptr)) len++;
      FCBlock** blockPtrs = new FCBlock*[len];
      for (int j=0; j<len; j++, k=(FCBlock*)(k->next_fcb_ptr)) blockPtrs[j]=k;
      kuaipai(blockPtrs, op, len);
    
      if (op == 1) { // LS_S (order by size)
        printf("===sort by file size===\n");
        // descending order
        if (len!=0) {
          for (int i=len-1; i>=0; i--) {
            printf("%s %d", blockPtrs[i]->name, blockPtrs[i]->size & 0x7FFF);
            if ((blockPtrs[i]->size & 0x8000) == 0x8000) printf(" d");  // dir judgement
            printf("\n");
          }
        }
      }

      else if (op == 0) { // LS_D (order by mod_time)
        printf("===sort by modified time===\n");
        // descending order
        if (len!=0) {
          for (int i=len-1; i>=0; i--) {
            printf("%s", blockPtrs[i]->name);
            if ((blockPtrs[i]->size & 0x8000) == 0x8000) printf(" d");  // dir judgement
            printf("\n");
          }
        }
      }

      delete [] blockPtrs;
    }
    else {
      if (op == 1) printf("===sort by file size===\n");
      else printf("===sort by modified time===\n");
    }

  }
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
  if (op == RM) { // RM

    FCBlock* rm_ptr, * FCB_stt_ptr = (FCBlock*)(fs->volume + fs->SUPERBLOCK_SIZE), * trace_currdir_ptr;
    
    trace_currdir_ptr = (FCBlock*)(curr_dir->subdir_ptr);

    // trace the file
    while (trace_currdir_ptr->next_fcb_ptr != 0) { 
      if (!(dev_strcmp(trace_currdir_ptr->name, s))) {rm_ptr = trace_currdir_ptr; break;}
      trace_currdir_ptr = (FCBlock*)(trace_currdir_ptr->next_fcb_ptr);
    }
    if (!(dev_strcmp(trace_currdir_ptr->name, s))) rm_ptr = trace_currdir_ptr;

    uchar* file_ptr = (fs->volume + fs->FILE_BASE_ADDRESS)+32*((rm_ptr->file_index & 0xFFFE) >> 1);

    // fcb directory logical structure update
    FCBlock* nxt=(FCBlock*)(rm_ptr->next_fcb_ptr);
    
    if (nxt == 0) {
	    if ((FCBlock*)(rm_ptr->prev_fcb_ptr) != curr_dir)  ((FCBlock*)(rm_ptr->prev_fcb_ptr))->next_fcb_ptr = 0;
  	  else  ((FCBlock*)(rm_ptr->prev_fcb_ptr))->subdir_ptr = 0;
    }

    else {
      ((FCBlock*)(rm_ptr->next_fcb_ptr))->prev_fcb_ptr = rm_ptr->prev_fcb_ptr;
      if ( !dev_strcmp( ((FCBlock*)(rm_ptr->prev_fcb_ptr))->name, curr_dir->name ) ) ((FCBlock*)(rm_ptr->prev_fcb_ptr))->subdir_ptr = rm_ptr->next_fcb_ptr;
      else ((FCBlock*)(rm_ptr->prev_fcb_ptr))->next_fcb_ptr = rm_ptr->next_fcb_ptr;
    }

    FCBlock* tt;
    // reallocate free space
    int deallo_block_num=(dev_abs(rm_ptr->size&0x7FFF)-1)/32 + 1;

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
    
    fs->file_curr_ptr = fs->file_curr_ptr - 32 * deallo_block_num;
    for (int i = 0; i < 32 * deallo_block_num; i++) (file_ptr)[i] = 0;

    // FCB space compaction
    for (FCBlock* i = rm_ptr+1; i <= fs->FCB_curr_ptr-1; i++) { 
      // ending of sub-dir consideration
      if (i->next_fcb_ptr != 0)  ((FCBlock*)(i->next_fcb_ptr))->prev_fcb_ptr = (u32*)((FCBlock*)(((FCBlock*)(i->next_fcb_ptr))->prev_fcb_ptr) - 1);

      // starting sub-dir consideration
      if ( !dev_strcmp( ((FCBlock*)(i->prev_fcb_ptr))->name, curr_dir->name ) ) {
        ((FCBlock*)(i->prev_fcb_ptr))->subdir_ptr = (u32*)((FCBlock*)(((FCBlock*)(i->prev_fcb_ptr))->subdir_ptr) - 1);
      }
      else {
        u32* prvnxt = ((FCBlock*)(i->prev_fcb_ptr))->next_fcb_ptr;
        ((FCBlock*)(i->prev_fcb_ptr))->next_fcb_ptr = (u32*)(((FCBlock*)prvnxt) - 1); 
      }
      
      *(i-1) = *i;
    }

    // nullify curr FCB
    fs->FCB_curr_ptr = fs->FCB_curr_ptr - 1;
    (fs->FCB_curr_ptr)->size=0;
    (fs->FCB_curr_ptr)->file_index=0;
    (fs->FCB_curr_ptr)->create_cnt=0;
    (fs->FCB_curr_ptr)->modify_cnt=0;
    (fs->FCB_curr_ptr)->next_fcb_ptr=0;
    (fs->FCB_curr_ptr)->prev_fcb_ptr=0;
    (fs->FCB_curr_ptr)->subdir_ptr=0;
    for (int i = 0; i <= 19; i++) (fs->FCB_curr_ptr)->name[i]=0;

    // fcb file index updates
    for (FCBlock* i = fs->FCB_curr_ptr-1; i >= rm_ptr; i--) {
      i->file_index = (((i->file_index>>1) - deallo_block_num)<<1) + 1;
    }

    // update dir size
    curr_dir->size = curr_dir->size - dev_strlen(s);
    curr_dir->modify_cnt = gtime++;
  }

  else if (op == RM_RF) {
    // Do DFS to rm all files and dirs inside the dir (in while loop maybe)
    FCBlock* k=(FCBlock*)(curr_dir->subdir_ptr);
    for (k; k->next_fcb_ptr!=0; k=(FCBlock*)(k->next_fcb_ptr)) {
      if (!dev_strcmp(k->name, s)) break;
    }

    dfs_rm(fs, k);
  }
  
  else if (op == CD) {
    // change curr dir and parent dir
    FCBlock* k=(FCBlock*)(curr_dir->subdir_ptr);

    for (k; k->next_fcb_ptr!=0; k=(FCBlock*)(k->next_fcb_ptr))
    {
      if (!dev_strcmp(k->name, s)) break;
    }
    
    par_dir=curr_dir;
    curr_dir=k;

    // change pwd
    int i=0;
    for (i; i<=3; i++) {
      if (pwd[i] == 0) break;
    }

    pwd[i]=k;
  }
  
  else if (op == MKDIR) {
    // create the new dir Physically
    FCBlock* fcbDir=fs->FCB_curr_ptr;
    int i;
    for (i = 0; s[i]!='\0'; i++) fcbDir->name[i] = s[i];
    fcbDir->name[i] = '\0';
    fcbDir->modify_cnt = gtime++;
    fcbDir->create_cnt = fcbDir->modify_cnt;
    fcbDir->size = 0;
    fcbDir->size = fcbDir->size + (1<<15); // is_dir update
    fcbDir->file_index = 0;
    fcbDir->file_index = fcbDir->file_index + 1; // valid bit update
    fs->FCB_curr_ptr = fs->FCB_curr_ptr+1;

    // k is at the last fcb node of the dir, 
    // create the new dir logically
    FCBlock* k=(FCBlock*)(curr_dir->subdir_ptr);
    for (k; k->next_fcb_ptr!=0; k=(FCBlock*)(k->next_fcb_ptr)) continue;
    k->next_fcb_ptr = (u32*)fcbDir;
    fcbDir->prev_fcb_ptr = (u32*)k;
    fcbDir->next_fcb_ptr = 0;
    fcbDir->subdir_ptr = 0;

    // update dir size
    curr_dir->size = curr_dir->size + dev_strlen(s);
    curr_dir->modify_cnt = gtime++;
  }
}
















