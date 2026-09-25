#pragma once
#include <stddef.h>
#include <stdint.h>

/* Owns fd and a copy of path after successful open. Copies may only be borrowed views;
 * only the owner closes it, after every read using it has finished. Reads use pread and never share a
 * mutable file position. This baseline uses buffered, synchronous I/O. */
typedef struct {
    int fd;
    uint64_t size;
    char* path;
} LmFile;

int lm_file_open(LmFile* file, const char* path);
void lm_file_close(LmFile* file);
/* Success means every requested byte was read. On failure the destination
 * may be partially modified and must never be consumed as tensor data. */
int lm_file_read(const LmFile* file, uint64_t offset, void* dest, size_t bytes);
