#pragma once
/* Legacy Win32 storage path, extracted without semantic changes.
 * The native POSIX baseline is in runtime_io_posix.h. */
/*
 * Load a tensor's data from GGUF via explicit unbuffered I/O
 * Returns allocated buffer (caller must free)
 */
/*
 * Read tensor data from GGUF via explicit I/O.
 * Returns a malloc'd buffer (caller can safely free()).
 * The read uses aligned I/O internally, then copies to a clean buffer.
 */
static void* read_tensor_from_handle(HANDLE hFile, uint64_t data_start,
                                     TensorInfo* tensor) {
    uint64_t offset = data_start + tensor->offset;
    uint64_t aligned = (offset / ALIGN) * ALIGN;
    int sub = (int)(offset - aligned);
    int read_size = (int)tensor->data_size + sub + ALIGN;
    read_size = ((read_size + ALIGN - 1) / ALIGN) * ALIGN;

    void* aligned_buf = _aligned_malloc(read_size, ALIGN);
    if (!aligned_buf) return NULL;

    LARGE_INTEGER li;
    li.QuadPart = aligned;
    SetFilePointerEx(hFile, li, NULL, FILE_BEGIN);
    DWORD br;
    ReadFile(hFile, aligned_buf, read_size, &br, NULL);

    /* Copy tensor data to a regular malloc buffer (safely free-able) */
    void* result = malloc((size_t)tensor->data_size);
    if (result) {
        memcpy(result, (char*)aligned_buf + sub, (size_t)tensor->data_size);
    }
    _aligned_free(aligned_buf);
    return result;
}

/* Persistent shard file handles (opened once, used for all reads) */
static HANDLE g_shard_handles[MAX_SHARDS] = {0};
static int g_handles_open = 0;

/* Separate handles: sync for weight loading, async for expert streaming */
static HANDLE g_async_handles[MAX_SHARDS] = {0};

static void open_shard_handles(GGUFModel* model) {
    int i;
    for (i = 0; i < model->num_shards; i++) {
        /* Sync handle for weight loading */
        g_shard_handles[i] = CreateFileA(model->shard_paths[i],
            GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
            FILE_FLAG_NO_BUFFERING, NULL);
        if (g_shard_handles[i] == INVALID_HANDLE_VALUE) {
            g_shard_handles[i] = CreateFileA(model->shard_paths[i],
                GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
        }
        /* Async handle for overlapped expert reads */
        g_async_handles[i] = CreateFileA(model->shard_paths[i],
            GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
            FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, NULL);
        if (g_async_handles[i] == INVALID_HANDLE_VALUE) {
            g_async_handles[i] = g_shard_handles[i]; /* fallback to sync */
        }
    }
    g_handles_open = 1;
}

/* Async read: issue non-blocking read, returns OVERLAPPED for later wait */
typedef struct {
    OVERLAPPED ov;
    void* buf;          /* aligned read buffer */
    int buf_size;
    void* dest;         /* where to copy data */
    int data_size;
    int sub_offset;     /* offset within aligned buffer */
    int valid;          /* 1 if async op was issued */
} AsyncRead;

static void async_read_start(AsyncRead* ar, int shard, uint64_t abs_offset,
                              void* dest, int data_size, void* aligned_buf, int buf_size) {
    uint64_t aligned = (abs_offset / ALIGN) * ALIGN;
    ar->sub_offset = (int)(abs_offset - aligned);
    ar->buf = aligned_buf;
    ar->buf_size = buf_size;
    ar->dest = dest;
    ar->data_size = data_size;

    memset(&ar->ov, 0, sizeof(OVERLAPPED));
    ar->ov.Offset = (DWORD)(aligned & 0xFFFFFFFF);
    ar->ov.OffsetHigh = (DWORD)(aligned >> 32);
    ar->ov.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

    DWORD br;
    BOOL ok = ReadFile(g_async_handles[shard], aligned_buf, buf_size, &br, &ar->ov);
    ar->valid = 1;
    /* ReadFile returns FALSE with ERROR_IO_PENDING for async — that's expected */
}

static void async_read_wait(AsyncRead* ar) {
    if (!ar->valid) return;
    WaitForSingleObject(ar->ov.hEvent, INFINITE);
    /* Copy data from aligned buffer to destination */
    if (ar->dest) {
        memcpy(ar->dest, (char*)ar->buf + ar->sub_offset, ar->data_size);
    }
    CloseHandle(ar->ov.hEvent);
    ar->valid = 0;
}

static void close_shard_handles(GGUFModel* model) {
    int i;
    for (i = 0; i < model->num_shards; i++) {
        if (g_shard_handles[i] && g_shard_handles[i] != INVALID_HANDLE_VALUE)
            CloseHandle(g_shard_handles[i]);
        if (g_async_handles[i] && g_async_handles[i] != INVALID_HANDLE_VALUE
            && g_async_handles[i] != g_shard_handles[i])
            CloseHandle(g_async_handles[i]);
    }
    g_handles_open = 0;
}

/* Read raw bytes from a shard at an absolute offset (for expert reads) */
static int read_bytes_from_shard(int shard, uint64_t abs_offset, void* dest,
                                  int size, void* aligned_buf, int aligned_buf_size) {
    uint64_t aligned = (abs_offset / ALIGN) * ALIGN;
    int sub = (int)(abs_offset - aligned);

    LARGE_INTEGER li;
    li.QuadPart = aligned;
    SetFilePointerEx(g_shard_handles[shard], li, NULL, FILE_BEGIN);
    DWORD br;
    ReadFile(g_shard_handles[shard], aligned_buf, aligned_buf_size, &br, NULL);

    /* Copy just the needed bytes */
    if (dest) memcpy(dest, (char*)aligned_buf + sub, size);
    return (int)br;
}
