/* Inspect headers/ranges without allocating tensor payloads or requiring a GPU.
 * This also exposes the production parser to small synthetic regression files. */
#include "../engine/runtime/gguf_parser.h"
int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: linmoe-inspect MODEL.gguf\n");
        return 2;
    }
    GGUFModel* model = (GGUFModel*)calloc(1, sizeof(*model));
    if (!model) return 1;
    int status = parse_gguf_split(argv[1], model);
    if (!status) {
        printf("shards=%d tensors=%llu\n", model->num_shards,
               (unsigned long long)model->n_tensors);
        for (uint64_t i = 0; i < model->n_tensors; ++i) {
            TensorInfo* t = &model->tensors[i];
            printf("%s type=%d bytes=%llu shard=%d offset=%llu\n", t->name, t->type,
                   (unsigned long long)t->data_size, t->shard,
                   (unsigned long long)(model->shard_data_starts[t->shard] + t->offset));
        }
    }
    free(model);
    return status ? 1 : 0;
}
