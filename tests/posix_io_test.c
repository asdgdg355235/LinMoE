/* Exercise the production reader at unaligned/EOF/64-bit offsets, including
 * truncation after open. Files are sparse so the >4 GiB test uses little disk. */
#include "../engine/platform/posix_io.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); } } while (0)

int main(void) {
    char path[] = "/tmp/linmoe-io-XXXXXX";
    int writer = mkstemp(path);
    CHECK(writer >= 0);
    const char payload[] = "unaligned payload";
    const uint64_t big = (UINT64_C(1) << 32) + 3;
    CHECK(pwrite(writer, payload, sizeof(payload), 3) == sizeof(payload));
    CHECK(pwrite(writer, payload, sizeof(payload), (off_t)big) == sizeof(payload));
    LmFile file;
    CHECK(lm_file_open(&file, path) == 0);
    char out[sizeof(payload)];
    CHECK(lm_file_read(&file, 3, out, sizeof(out)) == 0);
    CHECK(!memcmp(payload, out, sizeof(out)));
    CHECK(lm_file_read(&file, big, out, sizeof(out)) == 0);
    CHECK(!memcmp(payload, out, sizeof(out)));
    CHECK(lm_file_read(&file, file.size, NULL, 0) == 0);
    CHECK(lm_file_read(&file, file.size, out, 1) != 0);
    CHECK(lm_file_read(&file, UINT64_MAX, out, 1) != 0);
    CHECK(lm_file_read(&file, 0, NULL, 1) != 0);
    /* Failed range validation must leave the destination untouched. */
    memset(out, 0x55, sizeof(out));
    CHECK(lm_file_read(&file, big, out, sizeof(out) + 1) != 0);
    for (size_t i = 0; i < sizeof(out); ++i) CHECK(out[i] == 0x55);
    CHECK(ftruncate(writer, (off_t)(big + 2)) == 0);
    CHECK(lm_file_read(&file, big, out, sizeof(out)) != 0);
    CHECK(lm_file_read(&file, 3, out, sizeof(out)) == 0);
    CHECK(!memcmp(payload, out, sizeof(out)));
    lm_file_close(&file);
    CHECK(lm_file_read(&file, 0, out, 1) != 0);
    lm_file_close(&file); /* Closing a released object is safe. */
    CHECK(close(writer) == 0);
    CHECK(unlink(path) == 0);
    CHECK(lm_file_open(&file, path) != 0);
    lm_file_close(&file);
    CHECK(lm_file_open(&file, "/dev/null") != 0);
    lm_file_close(&file);
    puts("POSIX I/O regression tests PASS");
    return 0;
}
