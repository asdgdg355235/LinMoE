#include "posix_io.h"
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

_Static_assert(sizeof(off_t) >= 8, "LinMoE requires 64-bit file offsets");

/* Report the original request and progress, so truncation or device errors
 * can be traced to a specific shard rather than becoming bad model output. */
static int read_error(const LmFile* file, uint64_t offset, size_t bytes,
                      size_t done, int error) {
    fprintf(stderr, "LinMoE: pread path=%s offset=%" PRIu64
            " bytes=%zu completed=%zu: %s\n",
            file->path ? file->path : "<closed>", offset, bytes, done,
            strerror(error));
    errno = error;
    return -1;
}

int lm_file_open(LmFile* file, const char* path) {
    /* Initialize before opening so callers can close every attempted slot. */
    *file = (LmFile){.fd = -1};
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        fprintf(stderr, "LinMoE: open path=%s: %s\n", path, strerror(errno));
        return -1;
    }
    struct stat st;
    int error = 0;
    if (fstat(fd, &st) != 0) error = errno;
    else if (!S_ISREG(st.st_mode) || st.st_size < 0) error = EINVAL;
    if (!error) {
        file->path = strdup(path);
        if (!file->path) error = ENOMEM;
    }
    if (error) {
        fprintf(stderr, "LinMoE: inspect path=%s: %s\n", path, strerror(error));
        close(fd);
        errno = error;
        return -1;
    }
    file->fd = fd;
    file->size = (uint64_t)st.st_size;
    return 0;
}

void lm_file_close(LmFile* file) {
    /* Do not retry close on EINTR on Linux: the descriptor is already released
     * and a retry could close an unrelated descriptor reused by another thread. */
    if (file->fd >= 0 && close(file->fd) != 0)
        fprintf(stderr, "LinMoE: close path=%s: %s\n", file->path, strerror(errno));
    free(file->path);
    *file = (LmFile){.fd = -1};
}

int lm_file_read(const LmFile* file, uint64_t offset, void* dest, size_t bytes) {
    /* Subtraction-based checks prevent wraparound before converting to off_t.
     * A later truncation is still caught by the short-read/EOF loop below. */
    if (file->fd < 0) return read_error(file, offset, bytes, 0, EBADF);
    if ((!dest && bytes) || offset > file->size ||
        bytes > file->size - offset || offset > INT64_MAX ||
        bytes > (uint64_t)INT64_MAX - offset)
        return read_error(file, offset, bytes, 0, EINVAL);
    size_t done = 0;
    while (done < bytes) {
        size_t count = bytes - done;
        /* Bounded calls also work on kernels with a smaller per-read limit. */
        if (count > 1024 * 1024 * 1024) count = 1024 * 1024 * 1024;
        ssize_t n = pread(file->fd, (char*)dest + done, count, (off_t)(offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return read_error(file, offset, bytes, done, n ? errno : EIO);
        done += (size_t)n;
    }
    return 0;
}
