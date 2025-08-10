#pragma once
#include <hls_math.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include <hls_task.h>
#include <ap_int.h>
#include <hls_burst_maxi.h>

static constexpr int dim = 4096;
static constexpr int hidden_dim = 11008;
static constexpr int n_layers = 32;
static constexpr int n_heads = 32;
static constexpr int n_kv_heads = 32;
static constexpr int vocab_size = 32000;
static constexpr int seq_len = 256;
static constexpr int GS = 32;

static constexpr int head_size = dim / n_heads;
static constexpr int kv_dim = dim * n_kv_heads / n_heads;
static constexpr int kv_mul = n_heads / n_kv_heads;

#define PACK_SIZE 16
#define PACK_SIZE_QUANTIZE 64
#define PACK_SIZE_SCALE 32

#define DATATYPE_WIDTH 32
#define QUANTIZE_WIDTH 8
#define SCALE_WIDTH 8

typedef float DataType;
typedef ap_uint<DATATYPE_WIDTH*PACK_SIZE> DataType_pack;
typedef hls::burst_maxi<DataType_pack> DataType_pack_maxi;

typedef ap_fixed<QUANTIZE_WIDTH, 1> DataType_quantize;
typedef ap_uint<QUANTIZE_WIDTH*PACK_SIZE_QUANTIZE> DataType_quantize_pack;
typedef hls::burst_maxi<DataType_quantize_pack> DataType_quantize_pack_maxi;

typedef ap_uint<SCALE_WIDTH> DataType_scale;
typedef ap_uint<SCALE_WIDTH*PACK_SIZE_SCALE> DataType_scale_pack;
typedef hls::burst_maxi<DataType_scale_pack> DataType_scale_pack_maxi;

namespace hls {
    inline DataType exp(DataType x) { return std::exp(x); }
    inline DataType rsqrt(DataType x) { return 1.0f / std::sqrt(x); }
}

#define PC 16
static constexpr int MATMUL_DEPTH = 512;
static constexpr int MATMUL_WIDTH = 1024;
static constexpr int MULTIATTENTION_DEPTH = 32;

static constexpr int HBM_WEIGHT_SIZE = n_layers*(3*hidden_dim*dim/PACK_SIZE_QUANTIZE/PC + 4*dim/PC*dim/PACK_SIZE_QUANTIZE)+vocab_size*dim/PACK_SIZE_QUANTIZE/PC + seq_len/4*n_layers/PACK_SIZE_QUANTIZE*DATATYPE_WIDTH*dim;

static constexpr int DDR_WEIGHT_SCALE_SIZE = n_layers*(3*hidden_dim*dim/PACK_SIZE_SCALE/GS + 4*dim*dim/PACK_SIZE_SCALE/GS)+vocab_size*dim/PACK_SIZE_SCALE/GS;

static constexpr int DDR_BUS_SIZE = dim/PACK_SIZE+seq_len*dim/PACK_SIZE/2*2+n_layers*dim/PACK_SIZE*2+dim/PACK_SIZE;

static constexpr int MASK_PACK = PACK_SIZE - 1;
static constexpr int MASK_PACK_QUANTIZE = PACK_SIZE_QUANTIZE - 1;
static constexpr int MASK_PACK_SCALE = PACK_SIZE_SCALE - 1;
static constexpr int GS_LOG2 = 31 - __builtin_clz(GS);
static constexpr int PACK_SIZE_LOG2 = 31 - __builtin_clz(PACK_SIZE);
static constexpr int PACK_SIZE_QUANTIZE_LOG2 = 31 - __builtin_clz(PACK_SIZE_QUANTIZE);
static constexpr int PACK_SIZE_SCALE_LOG2 = 31 - __builtin_clz(PACK_SIZE_SCALE);
static constexpr int HEAD_SIZE_LOG2 = 31 - __builtin_clz(head_size);
static constexpr int MATMUL_DEPTH_LOG2 = 31 - __builtin_clz(MATMUL_DEPTH);
static constexpr int MATMUL_WIDTH_LOG2 = 31 - __builtin_clz(MATMUL_WIDTH);
static constexpr int MULTIATTENTION_DEPTH_LOG2 = 31 - __builtin_clz(MULTIATTENTION_DEPTH);

#define LAZY 0
#define RUN 1

#define RMSNORM_ATT 1
#define QUANTIZE_RMSNORM_ATT 2
#define MATMUL_Q_GENERATION 3
#define MATMUL_K_GENERATION 4
#define MATMUL_V_GENERATION 5
#define ROPE_QK 6
#define MULTIATTENTION 7
#define QUANTIZE_MULTIATTENTION 8
#define MATMUL_ATTENTION_OUTPUT 9
#define ACCUMULATE_ATTENTION_OUTPUT 10
#define RMSNORM_FFN 11
#define QUANTIZE_RMSNORM_FFN 12
#define MATMUL_GATE 13
#define MATMUL_UP 14
#define SWIGLU 15
#define QUANTIZE_SWIGLU 16
#define MATMUL_DOWN 17
#define ACCUMULATE_FFN_OUTPUT 18
#define RMSNORM_FINAL 19
#define QUANTIZE_RMSNORM_FINAL 20
#define MATMUL_LOGITS 21

struct Controller {
    int run_accumulate;
    int run_softmax;
    int run_rmsnorm;
    int run_matmul;
    int run_rope;
    int run_quantize;
    int run_multiattention;
    int run_swiglu;

    int index_rmsnorm;
    int index_quantize;
    int index_matmul;
    int index_accumulate;

    int exit;
};

inline void control_llama2_init(Controller* controller, int layer) {
    controller->run_accumulate = LAZY;
    controller->run_softmax = LAZY;
    controller->run_rmsnorm = RUN;
    controller->run_matmul = LAZY;
    controller->run_rope = LAZY;
    controller->run_quantize = LAZY;
    controller->run_multiattention = LAZY;
    controller->run_swiglu = LAZY;
    controller->exit = LAZY;

    controller->index_rmsnorm = layer == n_layers ? RMSNORM_FINAL : RMSNORM_ATT;
}

static constexpr int DATATYPE_LOG2 = 31 - __builtin_clz(DATATYPE_WIDTH);
static constexpr int QUANTIZE_LOG2 = 31 - __builtin_clz(QUANTIZE_WIDTH);
static constexpr int SCALE_LOG2 = 31 - __builtin_clz(SCALE_WIDTH);
#define PKR(i) ((((i)+1)<<(DATATYPE_LOG2))-1),((i)<<(DATATYPE_LOG2))
#define PKR_Q(i) ((((i)+1)<<(QUANTIZE_LOG2))-1),((i)<<(QUANTIZE_LOG2))
#define PKR_S(i) ((((i)+1)<<(SCALE_LOG2))-1),((i)<<(SCALE_LOG2))

inline ap_uint<DATATYPE_WIDTH> to_bits(DataType val) {
    #pragma HLS INLINE
    ap_uint<DATATYPE_WIDTH> res;

    uint32_t bits = *reinterpret_cast<uint32_t*>(&val);
    res.range(DATATYPE_WIDTH-1, 0) = bits;
    return res;
}
inline ap_uint<QUANTIZE_WIDTH> to_bits_quantize(DataType_quantize val) {
    #pragma HLS INLINE
    ap_uint<QUANTIZE_WIDTH> res;

    uint8_t bits = *reinterpret_cast<uint8_t*>(&val);
    res.range(QUANTIZE_WIDTH-1, 0) = bits;
    return res;
}
inline ap_uint<SCALE_WIDTH> to_bits_scale(DataType_scale val) {
    #pragma HLS INLINE
    ap_uint<SCALE_WIDTH> res;

    uint32_t bits = *reinterpret_cast<uint32_t*>(&val);
    res.range(SCALE_WIDTH-1, 0) = bits;
    return res;
}

inline DataType from_bits(ap_uint<DATATYPE_WIDTH> val) {
    #pragma HLS INLINE
    DataType res;

    uint32_t bits = val.range(DATATYPE_WIDTH-1, 0);
    res = *reinterpret_cast<DataType*>(&bits);
    return res;
}
inline DataType_quantize from_bits_quantize(ap_uint<QUANTIZE_WIDTH> val) {
    #pragma HLS INLINE
    DataType_quantize res;

    uint8_t bits = val.range(QUANTIZE_WIDTH-1, 0);
    res = *reinterpret_cast<DataType_quantize*>(&bits);
    return res;
}
inline DataType_scale from_bits_scale(ap_uint<SCALE_WIDTH> val) {
    #pragma HLS INLINE
    DataType_scale res;
    uint32_t bits = val.range(SCALE_WIDTH-1, 0);
    res = *reinterpret_cast<DataType_scale*>(&bits);
    return res;
}

static constexpr int X_PACK_IN_BASE_OFFSET = 0;
static constexpr int X_PACK_OUT_BASE_OFFSET = 0;
static constexpr int FREQ_CIS_REAL_PACK_BUS_BASE_OFFSET = dim / PACK_SIZE;
static constexpr int FREQ_CIS_IMAG_PACK_BUS_BASE_OFFSET = FREQ_CIS_REAL_PACK_BUS_BASE_OFFSET + seq_len * dim / PACK_SIZE / 2;
static constexpr int WEIGHT_RMSNORM_ATT_PACK_BUS_BASE_OFFSET = FREQ_CIS_IMAG_PACK_BUS_BASE_OFFSET + seq_len * dim / PACK_SIZE / 2;
static constexpr int WEIGHT_RMSNORM_FFN_PACK_BUS_BASE_OFFSET = WEIGHT_RMSNORM_ATT_PACK_BUS_BASE_OFFSET + n_layers * dim / PACK_SIZE;
static constexpr int WEIGHT_RMSNORM_FINAL_PACK_BASE_OFFSET = WEIGHT_RMSNORM_FFN_PACK_BUS_BASE_OFFSET + n_layers * dim / PACK_SIZE;

static constexpr int STRIDE_WEIGHT_BUS = 3*hidden_dim*dim/PACK_SIZE_QUANTIZE/PC + 4*dim/PC*dim/PACK_SIZE_QUANTIZE;
static constexpr int MATMUL_Q_GENERATION_WEIGHT_OFFSET = 0;
static constexpr int MATMUL_K_GENERATION_WEIGHT_OFFSET = MATMUL_Q_GENERATION_WEIGHT_OFFSET + dim * (dim / PC / PACK_SIZE_QUANTIZE);
static constexpr int MATMUL_V_GENERATION_WEIGHT_OFFSET = MATMUL_K_GENERATION_WEIGHT_OFFSET + kv_dim * (dim / PC / PACK_SIZE_QUANTIZE);
static constexpr int MATMUL_O_GENERATION_WEIGHT_OFFSET = MATMUL_V_GENERATION_WEIGHT_OFFSET + kv_dim * (dim / PC / PACK_SIZE_QUANTIZE);
static constexpr int MATMUL_1_GENERATION_WEIGHT_OFFSET = MATMUL_O_GENERATION_WEIGHT_OFFSET + dim * (dim / PC / PACK_SIZE_QUANTIZE);
static constexpr int MATMUL_3_GENERATION_WEIGHT_OFFSET = MATMUL_1_GENERATION_WEIGHT_OFFSET + hidden_dim * (dim / PC / PACK_SIZE_QUANTIZE);
static constexpr int MATMUL_2_GENERATION_WEIGHT_OFFSET = MATMUL_3_GENERATION_WEIGHT_OFFSET + hidden_dim * (dim / PC / PACK_SIZE_QUANTIZE);
static constexpr int MATMUL_LOGITS_GENERATION_WEIGHT_OFFSET = 0;
static constexpr int KV_CACHE_PACK_BUS_BASE_OFFSET = n_layers*(3*hidden_dim*dim/PACK_SIZE_QUANTIZE/PC + 4*dim/PC*dim/PACK_SIZE_QUANTIZE) + vocab_size*dim/PACK_SIZE_QUANTIZE/PC;

static constexpr int STRIDE_SCALE_BUS = 3*hidden_dim*dim/PACK_SIZE_SCALE/GS + 4*dim*dim/PACK_SIZE_SCALE/GS;
static constexpr int MATMUL_Q_GENERATION_SCALE_OFFSET = 0;
static constexpr int MATMUL_K_GENERATION_SCALE_OFFSET = MATMUL_Q_GENERATION_SCALE_OFFSET + dim * (dim / GS / PACK_SIZE_SCALE);
static constexpr int MATMUL_V_GENERATION_SCALE_OFFSET = MATMUL_K_GENERATION_SCALE_OFFSET + kv_dim * (dim / GS / PACK_SIZE_SCALE);
static constexpr int MATMUL_O_GENERATION_SCALE_OFFSET = MATMUL_V_GENERATION_SCALE_OFFSET + kv_dim * (dim / GS / PACK_SIZE_SCALE);
static constexpr int MATMUL_1_GENERATION_SCALE_OFFSET = MATMUL_O_GENERATION_SCALE_OFFSET + dim * (dim / GS / PACK_SIZE_SCALE);
static constexpr int MATMUL_3_GENERATION_SCALE_OFFSET = MATMUL_1_GENERATION_SCALE_OFFSET + hidden_dim * (dim / GS / PACK_SIZE_SCALE);
static constexpr int MATMUL_2_GENERATION_SCALE_OFFSET = MATMUL_3_GENERATION_SCALE_OFFSET + hidden_dim * (dim / GS / PACK_SIZE_SCALE);
static constexpr int MATMUL_LOGITS_GENERATION_SCALE_OFFSET = 0;

struct MaxiOffset {
    int weight_rmsnorm_att_pack_base_offset;
    int weight_rmsnorm_ffn_pack_base_offset;
    int weight_rmsnorm_final_pack_base_offset;
    int freq_cis_real_pack_base_offset;
    int freq_cis_imag_pack_base_offset;
    int key_cache_pack_base_offset;
    int value_cache_pack_base_offset;
    int weight_q_pack_base_offset;
    int weight_k_pack_base_offset;
    int weight_v_pack_base_offset;
    int weight_o_pack_base_offset;
    int weight_1_pack_base_offset;
    int weight_3_pack_base_offset;
    int weight_2_pack_base_offset;
    int weight_logits_pack_base_offset;
    int scale_q_pack_base_offset;
    int scale_k_pack_base_offset;
    int scale_v_pack_base_offset;
    int scale_o_pack_base_offset;
    int scale_1_pack_base_offset;
    int scale_3_pack_base_offset;
    int scale_2_pack_base_offset;
    int scale_logits_pack_base_offset;
    MaxiOffset(
        int weight_rmsnorm_att_pack_base_offset,
        int weight_rmsnorm_ffn_pack_base_offset,
        int weight_rmsnorm_final_pack_base_offset,
        int freq_cis_real_pack_base_offset,
        int freq_cis_imag_pack_base_offset,
        int key_cache_pack_base_offset,
        int value_cache_pack_base_offset,
        int weight_q_pack_base_offset,
        int weight_k_pack_base_offset,
        int weight_v_pack_base_offset,
        int weight_o_pack_base_offset,
        int weight_1_pack_base_offset,
        int weight_3_pack_base_offset,
        int weight_2_pack_base_offset,
        int weight_logits_pack_base_offset,
        int scale_q_pack_base_offset,
        int scale_k_pack_base_offset,
        int scale_v_pack_base_offset,
        int scale_o_pack_base_offset,
        int scale_1_pack_base_offset,
        int scale_3_pack_base_offset,
        int scale_2_pack_base_offset,
        int scale_logits_pack_base_offset
    ):  weight_rmsnorm_att_pack_base_offset(weight_rmsnorm_att_pack_base_offset),
        weight_rmsnorm_ffn_pack_base_offset(weight_rmsnorm_ffn_pack_base_offset),
        weight_rmsnorm_final_pack_base_offset(weight_rmsnorm_final_pack_base_offset),
        freq_cis_real_pack_base_offset(freq_cis_real_pack_base_offset),
        freq_cis_imag_pack_base_offset(freq_cis_imag_pack_base_offset),
        key_cache_pack_base_offset(key_cache_pack_base_offset),
        value_cache_pack_base_offset(value_cache_pack_base_offset),
        weight_q_pack_base_offset(weight_q_pack_base_offset),
        weight_k_pack_base_offset(weight_k_pack_base_offset),
        weight_v_pack_base_offset(weight_v_pack_base_offset),
        weight_o_pack_base_offset(weight_o_pack_base_offset),
        weight_1_pack_base_offset(weight_1_pack_base_offset),
        weight_3_pack_base_offset(weight_3_pack_base_offset),
        weight_2_pack_base_offset(weight_2_pack_base_offset),
        weight_logits_pack_base_offset(weight_logits_pack_base_offset),
        scale_q_pack_base_offset(scale_q_pack_base_offset),
        scale_k_pack_base_offset(scale_k_pack_base_offset),
        scale_v_pack_base_offset(scale_v_pack_base_offset),
        scale_o_pack_base_offset(scale_o_pack_base_offset),
        scale_1_pack_base_offset(scale_1_pack_base_offset),
        scale_3_pack_base_offset(scale_3_pack_base_offset),
        scale_2_pack_base_offset(scale_2_pack_base_offset),
        scale_logits_pack_base_offset(scale_logits_pack_base_offset)
    {}
};

struct DecoderBuffer {
    DataType x              [dim];
    DataType xb             [dim];
    DataType xb2            [dim];
    DataType hb             [hidden_dim];
    DataType hb2            [hidden_dim];
    DataType hb3            [hidden_dim];
    DataType q              [dim];
    DataType k              [kv_dim];
    DataType q2             [dim];
    DataType k2             [kv_dim];
    DataType v              [kv_dim];
    DataType_quantize xq    [dim];
    DataType_quantize hq    [hidden_dim];
    DataType_scale xq_scale [dim >> GS_LOG2];
    DataType_scale hq_scale [hidden_dim >> GS_LOG2];
    DataType logits         [vocab_size];
};

struct WeightPort{
    DataType_pack_maxi          ddr_bus_0;
    DataType_scale_pack_maxi    scale_bus;
    DataType_quantize_pack_maxi weight_bus_1;
    DataType_quantize_pack_maxi weight_bus_2;
    DataType_quantize_pack_maxi weight_bus_3;
    DataType_quantize_pack_maxi weight_bus_4;
    DataType_quantize_pack_maxi weight_bus_5;
    DataType_quantize_pack_maxi weight_bus_6;
    DataType_quantize_pack_maxi weight_bus_7;
    DataType_quantize_pack_maxi weight_bus_8;
    DataType_quantize_pack_maxi weight_bus_9;
    DataType_quantize_pack_maxi weight_bus_10;
    DataType_quantize_pack_maxi weight_bus_11;
    DataType_quantize_pack_maxi weight_bus_12;
    DataType_quantize_pack_maxi weight_bus_13;
    DataType_quantize_pack_maxi weight_bus_14;
    DataType_quantize_pack_maxi weight_bus_15;
    DataType_quantize_pack_maxi weight_bus_16;
    WeightPort(
        DataType_pack_maxi          ddr_bus_0,
        DataType_scale_pack_maxi    scale_bus,
        DataType_quantize_pack_maxi weight_bus_1,
        DataType_quantize_pack_maxi weight_bus_2,
        DataType_quantize_pack_maxi weight_bus_3,
        DataType_quantize_pack_maxi weight_bus_4,
        DataType_quantize_pack_maxi weight_bus_5,
        DataType_quantize_pack_maxi weight_bus_6,
        DataType_quantize_pack_maxi weight_bus_7,
        DataType_quantize_pack_maxi weight_bus_8,
        DataType_quantize_pack_maxi weight_bus_9,
        DataType_quantize_pack_maxi weight_bus_10,
        DataType_quantize_pack_maxi weight_bus_11,
        DataType_quantize_pack_maxi weight_bus_12,
        DataType_quantize_pack_maxi weight_bus_13,
        DataType_quantize_pack_maxi weight_bus_14,
        DataType_quantize_pack_maxi weight_bus_15,
        DataType_quantize_pack_maxi weight_bus_16
    ):  ddr_bus_0(ddr_bus_0),
        scale_bus(scale_bus),
        weight_bus_1(weight_bus_1),
        weight_bus_2(weight_bus_2),
        weight_bus_3(weight_bus_3),
        weight_bus_4(weight_bus_4),
        weight_bus_5(weight_bus_5),
        weight_bus_6(weight_bus_6),
        weight_bus_7(weight_bus_7),
        weight_bus_8(weight_bus_8),
        weight_bus_9(weight_bus_9),
        weight_bus_10(weight_bus_10),
        weight_bus_11(weight_bus_11),
        weight_bus_12(weight_bus_12),
        weight_bus_13(weight_bus_13),
        weight_bus_14(weight_bus_14),
        weight_bus_15(weight_bus_15),
        weight_bus_16(weight_bus_16)
        {}
};
