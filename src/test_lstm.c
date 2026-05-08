// test_lstm.c — bidirectional LSTM primitive parity test.
//
// Loads tmp/lstm_fixture.gguf (one bidir LSTM) + tmp/lstm_fixture_in.bin
// (T x in_size fp32), runs the LSTM via a ggml graph on the chosen backend,
// writes (T, 2H) output as fp32 to the path given by --output. Compared
// offline against tmp/lstm_fixture_ref.bin (torch nn.LSTM reference).
//
// This prototypes the kt_lstm_direction() helper that will move into
// kittens-tts.c once parity is confirmed.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#ifdef KT_HAVE_METAL
#include "ggml-metal.h"
#endif

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define LOG(fmt, ...) fprintf(stderr, "[test_lstm] " fmt "\n", ##__VA_ARGS__)
#define DIE(fmt, ...) do { fprintf(stderr, "[test_lstm] FATAL: " fmt "\n", ##__VA_ARGS__); exit(1); } while (0)

// ----------------------------------------------------------------------------
// Build one direction of a bidirectional LSTM as a ggml subgraph.
//
// Inputs (all live tensors, already in the backend buffer):
//   x      ne=(in_size, T)   F32  - input sequence
//   W      ne=(in_size, 4H)  F32  - weight_ih (PyTorch ifgo gate order)
//   R      ne=(H,       4H)  F32  - weight_hh
//   b      ne=(4H,)          F32  - combined bias (bias_ih + bias_hh)
//   h0/c0  ne=(H,)           F32  - initial states (zeros for non-stateful use)
//
// Outputs the per-timestep hidden state with ne=(H, T). Must be appended
// to the cgraph so the per-timestep cpy nodes get scheduled.
// ----------------------------------------------------------------------------

static struct ggml_tensor * lstm_direction(
    struct ggml_context * ctx,
    struct ggml_cgraph  * gf,
    struct ggml_tensor  * x,        // (in_size, T)
    struct ggml_tensor  * W,        // (in_size, 4H)
    struct ggml_tensor  * R,        // (H, 4H)
    struct ggml_tensor  * b,        // (4H,)
    struct ggml_tensor  * h0,       // (H,)
    struct ggml_tensor  * c0,       // (H,)
    int H, int T, int reverse,
    const char * name_prefix)
{
    // Pre-compute Wx for all timesteps in one shot: (4H, T) = mul_mat(W, x)
    // ggml_mul_mat(W, x) with W ne=(in_size, 4H) and x ne=(in_size, T):
    //   common dim = in_size, result ne = (4H, T).
    struct ggml_tensor * Wx = ggml_mul_mat(ctx, W, x);
    Wx = ggml_add(ctx, Wx, b);   // broadcast (4H,) over T

    // Pre-allocate output (H, T). Will be filled per-timestep via ggml_cpy.
    struct ggml_tensor * out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
    {
        char nm[64];
        snprintf(nm, sizeof(nm), "%s_out", name_prefix);
        ggml_set_name(out, nm);
    }

    struct ggml_tensor * h_prev = h0;
    struct ggml_tensor * c_prev = c0;

    const size_t f_sz = sizeof(float);

    for (int step = 0; step < T; step++) {
        const int t = reverse ? (T - 1 - step) : step;

        // Wx_t = view of column t of Wx (4H,)
        struct ggml_tensor * Wx_t =
            ggml_view_1d(ctx, Wx, 4 * H, (size_t) t * 4 * H * f_sz);

        // Rh = R @ h_prev — (4H,)
        struct ggml_tensor * Rh = ggml_mul_mat(ctx, R, h_prev);

        // z = Wx_t + Rh
        struct ggml_tensor * z = ggml_add(ctx, Wx_t, Rh);

        // Split z into 4 H-sized chunks: i, f, g, o (PyTorch gate order)
        struct ggml_tensor * zi = ggml_view_1d(ctx, z, H, 0);
        struct ggml_tensor * zf = ggml_view_1d(ctx, z, H, 1ull * H * f_sz);
        struct ggml_tensor * zg = ggml_view_1d(ctx, z, H, 2ull * H * f_sz);
        struct ggml_tensor * zo = ggml_view_1d(ctx, z, H, 3ull * H * f_sz);

        struct ggml_tensor * gi = ggml_sigmoid(ctx, zi);
        struct ggml_tensor * g_f = ggml_sigmoid(ctx, zf);
        struct ggml_tensor * gg = ggml_tanh   (ctx, zg);
        struct ggml_tensor * go = ggml_sigmoid(ctx, zo);

        // c_t = f * c_prev + i * g
        struct ggml_tensor * fc = ggml_mul(ctx, g_f, c_prev);
        struct ggml_tensor * ig = ggml_mul(ctx, gi,  gg);
        struct ggml_tensor * c_t = ggml_add(ctx, fc, ig);

        // h_t = o * tanh(c_t)
        struct ggml_tensor * h_t = ggml_mul(ctx, go, ggml_tanh(ctx, c_t));

        // Copy h_t into out[:, t]. Build forward expand to ensure scheduling.
        struct ggml_tensor * dest =
            ggml_view_1d(ctx, out, H, (size_t) t * H * f_sz);
        struct ggml_tensor * cpy = ggml_cpy(ctx, h_t, dest);
        ggml_build_forward_expand(gf, cpy);

        h_prev = h_t;
        c_prev = c_t;
    }
    return out;
}

// ----------------------------------------------------------------------------
// Backend init
// ----------------------------------------------------------------------------

static ggml_backend_t backend_init(const char * name) {
    if (!name || !strcmp(name, "cpu")) {
        ggml_backend_t b = ggml_backend_cpu_init();
        if (!b) DIE("ggml_backend_cpu_init failed");
        return b;
    }
#ifdef KT_HAVE_METAL
    if (!strcmp(name, "metal")) {
        ggml_backend_t b = ggml_backend_metal_init();
        if (!b) DIE("ggml_backend_metal_init failed");
        return b;
    }
#endif
    DIE("unknown backend: %s", name);
    return NULL;
}

static int gguf_u32(const struct gguf_context * g, const char * key) {
    int idx = gguf_find_key(g, key);
    if (idx < 0) DIE("missing GGUF key: %s", key);
    return (int) gguf_get_val_u32(g, idx);
}

// ----------------------------------------------------------------------------
// CLI
// ----------------------------------------------------------------------------

int main(int argc, char ** argv) {
    const char * gguf_path  = "tmp/lstm_fixture.gguf";
    const char * input_path = "tmp/lstm_fixture_in.bin";
    const char * out_path   = NULL;
    const char * backend_nm = "cpu";

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--gguf")    && i+1<argc) gguf_path  = argv[++i];
        else if (!strcmp(argv[i], "--input")   && i+1<argc) input_path = argv[++i];
        else if (!strcmp(argv[i], "--output")  && i+1<argc) out_path   = argv[++i];
        else if (!strcmp(argv[i], "--backend") && i+1<argc) backend_nm = argv[++i];
        else { fprintf(stderr, "usage: %s --gguf X --input X --output X [--backend cpu|metal]\n", argv[0]); return 1; }
    }
    if (!out_path) { fprintf(stderr, "--output required\n"); return 1; }

    ggml_backend_t backend = backend_init(backend_nm);
    LOG("backend: %s", backend_nm);

    // ---- load GGUF metadata + weights ----
    struct ggml_context * ctx_w = NULL;
    struct gguf_init_params gp = { /*.no_alloc=*/ true, /*.ctx=*/ &ctx_w };
    struct gguf_context * gctx = gguf_init_from_file(gguf_path, gp);
    if (!gctx) DIE("gguf_init_from_file failed: %s", gguf_path);

    const int in_size = gguf_u32(gctx, "kittens-lstm-test.in_size");
    const int H       = gguf_u32(gctx, "kittens-lstm-test.hidden");
    const int T       = gguf_u32(gctx, "kittens-lstm-test.T");
    LOG("LSTM: in=%d H=%d T=%d", in_size, H, T);

    ggml_backend_buffer_t weights_buf = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    if (!weights_buf) DIE("alloc_ctx_tensors failed");

    // Stream weights from file into backend buffer
    {
        FILE * f = fopen(gguf_path, "rb");
        if (!f) DIE("fopen %s", gguf_path);
        const size_t base = gguf_get_data_offset(gctx);
        const int n = gguf_get_n_tensors(gctx);
        size_t max_nb = 0;
        for (int i = 0; i < n; i++) {
            struct ggml_tensor * t = ggml_get_tensor(ctx_w, gguf_get_tensor_name(gctx, i));
            if (ggml_nbytes(t) > max_nb) max_nb = ggml_nbytes(t);
        }
        void * staging = malloc(max_nb);
        for (int i = 0; i < n; i++) {
            const char * nm = gguf_get_tensor_name(gctx, i);
            struct ggml_tensor * t = ggml_get_tensor(ctx_w, nm);
            const size_t off = base + gguf_get_tensor_offset(gctx, i);
            const size_t nb  = ggml_nbytes(t);
            fseek(f, (long) off, SEEK_SET);
            if (fread(staging, 1, nb, f) != nb) DIE("short read %s", nm);
            ggml_backend_tensor_set(t, staging, 0, nb);
        }
        free(staging);
        fclose(f);
    }

    struct ggml_tensor * fwd_W = ggml_get_tensor(ctx_w, "lstm.fwd.W");
    struct ggml_tensor * fwd_R = ggml_get_tensor(ctx_w, "lstm.fwd.R");
    struct ggml_tensor * fwd_b = ggml_get_tensor(ctx_w, "lstm.fwd.b");
    struct ggml_tensor * bwd_W = ggml_get_tensor(ctx_w, "lstm.bwd.W");
    struct ggml_tensor * bwd_R = ggml_get_tensor(ctx_w, "lstm.bwd.R");
    struct ggml_tensor * bwd_b = ggml_get_tensor(ctx_w, "lstm.bwd.b");
    if (!fwd_W || !bwd_W) DIE("missing LSTM tensors in GGUF");

    // ---- read input ----
    FILE * fi = fopen(input_path, "rb");
    if (!fi) DIE("fopen %s", input_path);
    fseek(fi, 0, SEEK_END);
    long in_bytes = ftell(fi);
    fseek(fi, 0, SEEK_SET);
    if (in_bytes != (long)(sizeof(float) * in_size * T))
        DIE("input size mismatch: got %ld, expected %ld",
            in_bytes, (long)(sizeof(float) * in_size * T));
    float * x_host = malloc(in_bytes);
    if (fread(x_host, 1, in_bytes, fi) != (size_t) in_bytes) DIE("read input");
    fclose(fi);

    // ---- build graph ----
    const size_t graph_mem =
        ggml_tensor_overhead() * 65536 + ggml_graph_overhead_custom(65536, false);
    void * mem_buf = malloc(graph_mem);
    struct ggml_init_params ip = {
        /*.mem_size  =*/ graph_mem,
        /*.mem_buffer=*/ mem_buf,
        /*.no_alloc  =*/ true,
    };
    struct ggml_context * gf_ctx = ggml_init(ip);

    struct ggml_cgraph * gf = ggml_new_graph_custom(gf_ctx, 65536, false);

    // Inputs that we'll fill in step (5)
    struct ggml_tensor * x  = ggml_new_tensor_2d(gf_ctx, GGML_TYPE_F32, in_size, T);
    struct ggml_tensor * h0 = ggml_new_tensor_1d(gf_ctx, GGML_TYPE_F32, H);
    struct ggml_tensor * c0 = ggml_new_tensor_1d(gf_ctx, GGML_TYPE_F32, H);
    ggml_set_input(x);  ggml_set_input(h0);  ggml_set_input(c0);
    ggml_set_name(x, "x");

    struct ggml_tensor * fwd_out =
        lstm_direction(gf_ctx, gf, x, fwd_W, fwd_R, fwd_b, h0, c0, H, T, /*reverse=*/0, "fwd");
    struct ggml_tensor * bwd_out =
        lstm_direction(gf_ctx, gf, x, bwd_W, bwd_R, bwd_b, h0, c0, H, T, /*reverse=*/1, "bwd");

    // Concat along channel dim → (2H, T)
    struct ggml_tensor * out = ggml_concat(gf_ctx, fwd_out, bwd_out, /*dim=*/0);
    ggml_set_name(out, "out");
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    LOG("graph nodes=%d", ggml_graph_n_nodes(gf));

    // ---- allocate activations on backend ----
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) DIE("gallocr_alloc_graph failed");

    // ---- upload inputs ----
    ggml_backend_tensor_set(x, x_host, 0, in_bytes);
    {
        float * z = calloc(H, sizeof(float));
        ggml_backend_tensor_set(h0, z, 0, H * sizeof(float));
        ggml_backend_tensor_set(c0, z, 0, H * sizeof(float));
        free(z);
    }

    // ---- compute ----
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) DIE("compute failed");
    clock_gettime(CLOCK_MONOTONIC, &t1);
    const double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1.0e6;
    LOG("forward %.2f ms", ms);

    // ---- read output: ne=(2H, T) → buffer of 2H*T floats ----
    const size_t out_n = (size_t) 2 * H * T;
    float * y = malloc(out_n * sizeof(float));
    ggml_backend_tensor_get(out, y, 0, out_n * sizeof(float));

    FILE * fo = fopen(out_path, "wb");
    if (!fo) DIE("fopen %s", out_path);
    if (fwrite(y, sizeof(float), out_n, fo) != out_n) DIE("write output");
    fclose(fo);
    LOG("wrote %s (%zu f32)", out_path, out_n);

    free(y);
    free(x_host);
    ggml_gallocr_free(galloc);
    ggml_free(gf_ctx);
    free(mem_buf);
    ggml_backend_buffer_free(weights_buf);
    ggml_free(ctx_w);
    gguf_free(gctx);
    ggml_backend_free(backend);
    return 0;
}
