// Per-expert routing statistics for MoE GGUFs, collected through the llama.cpp eval callback.
//
// For every MoE layer it accumulates, per routed expert: routed-token counts, the sum of final
// (renormalized) routing weights, the sum and sum of squares of expert-output L2 norms, and the
// REAP numerator sum(g * ||f||). These are the same accumulators as moep.profiler, so
// moep.scoring criteria apply unchanged. Documents alternate between two halves so split-half
// reliability can be checked from one run.
//
// Input (-f): calibration text, documents separated by a line containing only DOC_SEP.
// Output (-o): JSON. Each document is truncated to its first -c tokens (one sequence per document).

#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"
#include "log.h"

#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

static const char * DOC_SEP = "<|moe_stats_doc|>";

struct half_stats {
    double tokens = 0;
    std::vector<double> counts, gate_sum, norm_sum, norm_sq_sum, reap_sum;

    void resize(int n) {
        for (auto * v : {&counts, &gate_sum, &norm_sum, &norm_sq_sum, &reap_sum}) {
            v->resize(n, 0.0);
        }
    }
};

struct layer_state {
    int n_expert = 0;
    half_stats halves[2];
    // tensors of the current ubatch, consumed when ffn_moe_down arrives
    std::vector<int32_t> topk;
    std::vector<float> weights;
    bool have_norm_weights = false;
    int n_used = 0;
    int n_tokens = 0;
};

struct collector {
    std::map<int, layer_state> layers;
    std::vector<uint8_t> buf;
    int half = 0;
};

static bool split_name(const char * name, std::string & prefix, int & il) {
    const char * dash = strrchr(name, '-');
    if (dash == nullptr) {
        return false;
    }
    prefix.assign(name, dash - name);
    il = atoi(dash + 1);
    return true;
}

static const uint8_t * host_data(ggml_tensor * t, collector & c) {
    if (ggml_backend_buffer_is_host(t->buffer)) {
        return (const uint8_t *) t->data;
    }
    c.buf.resize(ggml_nbytes(t));
    ggml_backend_tensor_get(t, c.buf.data(), 0, c.buf.size());
    return c.buf.data();
}

static float read_f(const uint8_t * p, ggml_type type) {
    return type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(*(const ggml_fp16_t *) p) : *(const float *) p;
}

static bool on_tensor(ggml_tensor * t, bool ask, void * user_data) {
    auto & c = *(collector *) user_data;
    std::string prefix;
    int il;
    if (!split_name(t->name, prefix, il)) {
        return false;
    }

    if (ask) {
        if (prefix == "ffn_moe_logits") {
            auto & ls = c.layers[il];
            if (ls.n_expert < t->ne[0]) {
                ls.n_expert = (int) t->ne[0];
                ls.halves[0].resize(ls.n_expert);
                ls.halves[1].resize(ls.n_expert);
            }
            return false;
        }
        return prefix == "ffn_moe_topk" || prefix == "ffn_moe_weights" || prefix == "ffn_moe_weights_norm" ||
               prefix == "ffn_moe_down";
    }

    auto & ls = c.layers[il];
    const uint8_t * d = host_data(t, c);

    if (prefix == "ffn_moe_topk") {
        GGML_ASSERT(t->type == GGML_TYPE_I32 && ls.n_expert > 0);
        ls.n_used   = (int) t->ne[0];
        ls.n_tokens = (int) t->ne[1];
        ls.topk.resize((size_t) ls.n_used * ls.n_tokens);
        for (int tok = 0; tok < ls.n_tokens; ++tok) {
            for (int j = 0; j < ls.n_used; ++j) {
                ls.topk[(size_t) tok * ls.n_used + j] = *(const int32_t *) (d + j * t->nb[0] + tok * t->nb[1]);
            }
        }
        ls.have_norm_weights = false;
        return true;
    }

    if (prefix == "ffn_moe_weights" || prefix == "ffn_moe_weights_norm") {
        // raw weights are [1, n_used, n_tokens]; renormalized ones are [n_used, n_tokens]
        if (prefix == "ffn_moe_weights" && ls.have_norm_weights) {
            return true;
        }
        const bool lead_one = t->ne[0] == 1;
        const size_t nb_j   = lead_one ? t->nb[1] : t->nb[0];
        const size_t nb_tok = lead_one ? t->nb[2] : t->nb[1];
        ls.weights.resize((size_t) ls.n_used * ls.n_tokens);
        for (int tok = 0; tok < ls.n_tokens; ++tok) {
            for (int j = 0; j < ls.n_used; ++j) {
                ls.weights[(size_t) tok * ls.n_used + j] = read_f(d + j * nb_j + tok * nb_tok, t->type);
            }
        }
        ls.have_norm_weights = prefix == "ffn_moe_weights_norm";
        return true;
    }

    // ffn_moe_down: [n_embd_expert, n_used, n_tokens], one output per selected expert
    GGML_ASSERT(t->ne[1] == ls.n_used && t->ne[2] == ls.n_tokens);
    auto & h = ls.halves[c.half];
    for (int tok = 0; tok < ls.n_tokens; ++tok) {
        for (int j = 0; j < ls.n_used; ++j) {
            const uint8_t * row = d + j * t->nb[1] + tok * t->nb[2];
            double sq = 0.0;
            if (t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float)) {
                const float * x = (const float *) row;
                float acc = 0.0f;
                for (int64_t i = 0; i < t->ne[0]; ++i) {
                    acc += x[i] * x[i];
                }
                sq = acc;
            } else {
                for (int64_t i = 0; i < t->ne[0]; ++i) {
                    const double x = read_f(row + i * t->nb[0], t->type);
                    sq += x * x;
                }
            }
            const double n = std::sqrt(sq);
            const int    e = ls.topk[(size_t) tok * ls.n_used + j];
            const double g = ls.weights[(size_t) tok * ls.n_used + j];
            GGML_ASSERT(e >= 0 && e < ls.n_expert);
            h.counts[e] += 1.0;
            h.gate_sum[e] += g;
            h.norm_sum[e] += n;
            h.norm_sq_sum[e] += sq;
            h.reap_sum[e] += g * n;
        }
    }
    h.tokens += ls.n_tokens;
    return true;
}

static void write_array(FILE * f, const std::vector<double> & v) {
    fputc('[', f);
    for (size_t i = 0; i < v.size(); ++i) {
        fprintf(f, i ? ",%.9g" : "%.9g", v[i]);
    }
    fputc(']', f);
}

static void write_json(const collector & c, const std::string & path, const std::string & model, size_t n_seqs,
                       size_t n_tok) {
    FILE * f = fopen(path.c_str(), "w");
    GGML_ASSERT(f != nullptr);
    fprintf(f, "{\"model\":\"%s\",\"n_seqs\":%zu,\"n_tokens\":%zu,\"layers\":{", model.c_str(), n_seqs, n_tok);
    bool first = true;
    for (const auto & [il, ls] : c.layers) {
        if (ls.halves[0].tokens + ls.halves[1].tokens == 0) {
            continue;
        }
        fprintf(f, "%s\"%d\":{\"n_expert\":%d,\"halves\":[", first ? "" : ",", il, ls.n_expert);
        for (int hi = 0; hi < 2; ++hi) {
            const auto & h = ls.halves[hi];
            fprintf(f, "%s{\"tokens\":%.0f,\"counts\":", hi ? "," : "", h.tokens);
            write_array(f, h.counts);
            fputs(",\"gate_sum\":", f);
            write_array(f, h.gate_sum);
            fputs(",\"norm_sum\":", f);
            write_array(f, h.norm_sum);
            fputs(",\"norm_sq_sum\":", f);
            write_array(f, h.norm_sq_sum);
            fputs(",\"reap_sum\":", f);
            write_array(f, h.reap_sum);
            fputc('}', f);
        }
        fputs("]}", f);
        first = false;
    }
    fputs("}}\n", f);
    fclose(f);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();
    // the imatrix example type registers the -f and -o options this tool needs
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_IMATRIX)) {
        return 1;
    }
    if (params.prompt.empty()) {
        LOG_ERR("usage: %s -m model.gguf -f calib.txt -o stats.json -c <max tokens per doc> -b <same>\n", argv[0]);
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    collector c;
    params.cb_eval           = on_tensor;
    params.cb_eval_user_data = &c;
    params.warmup            = false;

    auto llama_init = common_init_from_params(params);
    auto * model    = llama_init->model();
    auto * ctx      = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init\n", __func__);
        return 1;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const bool          add_bos = llama_vocab_get_add_bos(vocab);
    const int           max_tokens = std::min(params.n_ctx, params.n_batch);

    std::vector<std::string> docs;
    size_t pos = 0;
    const std::string sep = std::string("\n") + DOC_SEP + "\n";
    while (pos <= params.prompt.size()) {
        const size_t next = params.prompt.find(sep, pos);
        const std::string doc = params.prompt.substr(pos, next == std::string::npos ? std::string::npos : next - pos);
        if (!doc.empty()) {
            docs.push_back(doc);
        }
        if (next == std::string::npos) {
            break;
        }
        pos = next + sep.size();
    }

    size_t n_seqs = 0, n_tok = 0;
    for (size_t di = 0; di < docs.size(); ++di) {
        auto tokens = common_tokenize(ctx, docs[di], add_bos, true);
        if (tokens.size() > (size_t) max_tokens) {
            tokens.resize(max_tokens);
        }
        if (tokens.size() >= 2) {
            llama_memory_clear(llama_get_memory(ctx), true);
            c.half = (int) (di % 2);
            common_batch batch = common_batch_get_one(ctx, tokens);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
                LOG_ERR("%s: decode failed on doc %zu\n", __func__, di);
                return 1;
            }
            ++n_seqs;
            n_tok += tokens.size();
        }
        if ((di + 1) % 50 == 0 || di + 1 == docs.size()) {
            LOG_INF("moe_stats: %zu/%zu docs, %zu tokens\n", di + 1, docs.size(), n_tok);
            write_json(c, params.out_file, params.model.path, n_seqs, n_tok);
        }
    }

    write_json(c, params.out_file, params.model.path, n_seqs, n_tok);
    LOG_INF("moe_stats: wrote %s (%zu tokens, %zu MoE layers)\n", params.out_file.c_str(), n_tok, c.layers.size());
    llama_backend_free();
    return 0;
}
