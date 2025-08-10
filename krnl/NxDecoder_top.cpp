#include "../include/decoder_top.h"

void load_decoder_buffer_x(
    DataType* x,
    DataType_pack_maxi x_pack,
    int offset
) {
    #pragma HLS inline off

    const int pack_len = dim/PACK_SIZE;

    const int req_size = pack_len / 64;
    load_decoder_buffer_x_loop1:
    for(int i = 0; i < req_size; i ++) {
        #pragma HLS PIPELINE II=1
        x_pack.read_request(offset+i*64, 64);
    }

    DataType_pack pack;
    load_decoder_buffer_x_loop2:
    for(int i = 0; i < dim; i ++) {
        #pragma HLS PIPELINE II=1
        if(i % PACK_SIZE == 0) pack = x_pack.read();
        const int offset = i & MASK_PACK;
        x[i] = from_bits(pack(PKR(offset)));
    }
}

void save_decoder_buffer_x(
    DataType* x,
    DataType_pack_maxi x_pack,
    int offset
) {
    #pragma HLS inline off

    const int pack_len = dim/PACK_SIZE;

    const int req_size = pack_len / 64;
    save_decoder_buffer_x_loop1:
    for(int i = 0; i < req_size; i ++) {
        #pragma HLS PIPELINE II=1
        x_pack.write_request(offset+i*64, 64);
    }

    DataType_pack pack;
    save_decoder_buffer_x_loop2:
    for(int i = 0; i < dim; i ++) {
        #pragma HLS PIPELINE II=1
        const int offset = i & MASK_PACK;
        pack(PKR(offset)) = to_bits(x[i]);
        if(i % PACK_SIZE == PACK_SIZE - 1) {
            x_pack.write(pack);
        }
    }
    save_decoder_buffer_x_loop3:
    for(int i = 0; i < req_size; i ++) {
        #pragma HLS PIPELINE II=1
        x_pack.write_response();
    }
}

extern "C" void NxDecoder_top(
    int pos,
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
) {

    #pragma HLS INTERFACE mode=m_axi offset=slave max_widen_bitwidth=256 max_read_burst_length=1  num_read_outstanding=256 max_write_burst_length=1  num_write_outstanding=1 bundle=scale_bus     port=scale_bus     depth=DDR_WEIGHT_SCALE_SIZE
    #pragma HLS INTERFACE mode=m_axi offset=slave max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=4   max_write_burst_length=64 num_write_outstanding=4 bundle=ddr_bus0      port=ddr_bus_0     depth=DDR_BUS_SIZE

    #pragma HLS INTERFACE mode=m_axi offset=slave max_widen_bitwidth=512 max_read_burst_length=1  num_read_outstanding=256 max_write_burst_length=1  num_write_outstanding=1  bundle=weight_bus1  port=weight_bus_1  depth=HBM_WEIGHT_SIZE
    #pragma HLS INTERFACE mode=m_axi offset=slave max_widen_bitwidth=512 max_read_burst_length=1  num_read_outstanding=256 max_write_burst_length=1  num_write_outstanding=1  bundle=weight_bus2  port=weight_bus_2  depth=HBM_WEIGHT_SIZE
    #pragma HLS INTERFACE mode=m_axi offset=slave max_widen_bitwidth=512 max_read_burst_length=1  num_read_outstanding=256 max_write_burst_length=1  num_write_outstanding=1  bundle=weight_bus3  port=weight_bus_3  depth=HBM_WEIGHT_SIZE
    #pragma HLS INTERFACE mode=m_axi offset=slave max_widen_bitwidth=512 max_read_burst_length=1  num_read_outstanding=256 max_write_burst_length=1  num_write_outstanding=1  bundle=weight_bus4  port=weight_bus_4  depth=HBM_WEIGHT_SIZE
    #pragma HLS INTERFACE mode=m_axi offset=slave max_widen_bitwidth=512 max_read_burst_length=1  num_read_outstanding=256 max_write_burst_length=1  num_write_outstanding=1  bundle=weight_bus5  port=weight_bus_5  depth=HBM_WEIGHT_SIZE
    #pragma HLS INTERFACE mode=m_axi offset=slave max_widen_bitwidth=512 max_read_burst_length=1  num_read_outstanding=256 max_write_burst_length=1  num_write_outstanding=1  bundle=weight_bus6  port=weight_bus_6  depth=HBM_WEIGHT_SIZE
    #pragma HLS INTERFACE mode=m_axi offset=slave max_widen_bitwidth=512 max_read_burst_length=1  num_read_outstanding=256 max_write_burst_length=1  num_write_outstanding=1  bundle=weight_bus7  port=weight_bus_7  depth=HBM_WEIGHT_SIZE
    #pragma HLS INTERFACE mode=m_axi offset=slave max_widen_bitwidth=512 max_read_burst_length=1  num_read_outstanding=256 max_write_burst_length=1  num_write_outstanding=1  bundle=weight_bus8  port=weight_bus_8  depth=HBM_WEIGHT_SIZE
    #pragma HLS INTERFACE mode=m_axi offset=slave max_widen_bitwidth=512 max_read_burst_length=1  num_read_outstanding=256 max_write_burst_length=1  num_write_outstanding=1  bundle=weight_bus9  port=weight_bus_9  depth=HBM_WEIGHT_SIZE
    #pragma HLS INTERFACE mode=m_axi offset=slave max_widen_bitwidth=512 max_read_burst_length=1  num_read_outstanding=256 max_write_burst_length=1  num_write_outstanding=1  bundle=weight_bus10 port=weight_bus_10 depth=HBM_WEIGHT_SIZE
    #pragma HLS INTERFACE mode=m_axi offset=slave max_widen_bitwidth=512 max_read_burst_length=1  num_read_outstanding=256 max_write_burst_length=1  num_write_outstanding=1  bundle=weight_bus11 port=weight_bus_11 depth=HBM_WEIGHT_SIZE
    #pragma HLS INTERFACE mode=m_axi offset=slave max_widen_bitwidth=512 max_read_burst_length=1  num_read_outstanding=256 max_write_burst_length=1  num_write_outstanding=1  bundle=weight_bus12 port=weight_bus_12 depth=HBM_WEIGHT_SIZE
    #pragma HLS INTERFACE mode=m_axi offset=slave max_widen_bitwidth=512 max_read_burst_length=1  num_read_outstanding=256 max_write_burst_length=1  num_write_outstanding=1  bundle=weight_bus13 port=weight_bus_13 depth=HBM_WEIGHT_SIZE
    #pragma HLS INTERFACE mode=m_axi offset=slave max_widen_bitwidth=512 max_read_burst_length=1  num_read_outstanding=256 max_write_burst_length=1  num_write_outstanding=1  bundle=weight_bus14 port=weight_bus_14 depth=HBM_WEIGHT_SIZE
    #pragma HLS INTERFACE mode=m_axi offset=slave max_widen_bitwidth=512 max_read_burst_length=1  num_read_outstanding=256 max_write_burst_length=1  num_write_outstanding=1  bundle=weight_bus15 port=weight_bus_15 depth=HBM_WEIGHT_SIZE
    #pragma HLS INTERFACE mode=m_axi offset=slave max_widen_bitwidth=512 max_read_burst_length=1  num_read_outstanding=256 max_write_burst_length=1  num_write_outstanding=1  bundle=weight_bus16 port=weight_bus_16 depth=HBM_WEIGHT_SIZE

    DecoderBuffer decoder_buffer;
    #pragma HLS bind_storage variable=decoder_buffer.logits type=RAM_2P impl=uram
    #pragma HLS bind_storage variable=decoder_buffer.hb type=RAM_2P impl=uram
    #pragma HLS bind_storage variable=decoder_buffer.hb2 type=RAM_2P impl=uram
    #pragma HLS bind_storage variable=decoder_buffer.hb3 type=RAM_2P impl=uram
    #pragma HLS bind_storage variable=decoder_buffer.x type=RAM_2P impl=uram
    #pragma HLS bind_storage variable=decoder_buffer.xb type=RAM_2P impl=uram
    #pragma HLS bind_storage variable=decoder_buffer.xb2 type=RAM_2P impl=uram
    #pragma HLS bind_storage variable=decoder_buffer.q type=RAM_2P impl=uram
    #pragma HLS bind_storage variable=decoder_buffer.k type=RAM_2P impl=uram
    #pragma HLS bind_storage variable=decoder_buffer.v type=RAM_2P impl=uram
    #pragma HLS bind_storage variable=decoder_buffer.q2 type=RAM_2P impl=uram
    #pragma HLS bind_storage variable=decoder_buffer.k2 type=RAM_2P impl=uram
    #pragma HLS bind_storage variable=decoder_buffer.xq type=RAM_2P impl=uram
    #pragma HLS bind_storage variable=decoder_buffer.hq type=RAM_2P impl=uram
    #pragma HLS bind_storage variable=decoder_buffer.xq_scale type=RAM_2P impl=uram
    #pragma HLS bind_storage variable=decoder_buffer.hq_scale type=RAM_2P impl=uram
    #pragma HLS ARRAY_RESHAPE variable=decoder_buffer.xq dim=1 cyclic factor=32
    #pragma HLS ARRAY_RESHAPE variable=decoder_buffer.hq dim=1 cyclic factor=32
    #pragma HLS ARRAY_RESHAPE variable=decoder_buffer.q dim=1 cyclic factor=32
    #pragma HLS ARRAY_RESHAPE variable=decoder_buffer.k dim=1 cyclic factor=32
    #pragma HLS ARRAY_RESHAPE variable=decoder_buffer.v dim=1 cyclic factor=32

    #pragma HLS ARRAY_RESHAPE variable=decoder_buffer.logits dim=1 cyclic factor=32
    #pragma HLS ARRAY_RESHAPE variable=decoder_buffer.hb dim=1 cyclic factor=32
    #pragma HLS ARRAY_RESHAPE variable=decoder_buffer.hb2 dim=1 cyclic factor=32
    #pragma HLS ARRAY_RESHAPE variable=decoder_buffer.xb dim=1 cyclic factor=32

    #pragma HLS ARRAY_RESHAPE variable=decoder_buffer.xb2 dim=1 cyclic factor=32

    WeightPort weight_port(
        ddr_bus_0,
        scale_bus,
        weight_bus_1,
        weight_bus_2,
        weight_bus_3,
        weight_bus_4,
        weight_bus_5,
        weight_bus_6,
        weight_bus_7,
        weight_bus_8,
        weight_bus_9,
        weight_bus_10,
        weight_bus_11,
        weight_bus_12,
        weight_bus_13,
        weight_bus_14,
        weight_bus_15,
        weight_bus_16
    );

    load_decoder_buffer_x(decoder_buffer.x, weight_port.ddr_bus_0, X_PACK_IN_BASE_OFFSET);

    NxDecoder_loop:
    for(int i = 0; i <= n_layers; i ++) {
        #pragma HLS UNROLL off=true
        int weight_rmsnorm_att_pack_base_offset = WEIGHT_RMSNORM_ATT_PACK_BUS_BASE_OFFSET + i * (dim / PACK_SIZE);
        int weight_rmsnorm_ffn_pack_base_offset = WEIGHT_RMSNORM_FFN_PACK_BUS_BASE_OFFSET + i * (dim / PACK_SIZE);
        int freq_cis_real_pack_base_offset = FREQ_CIS_REAL_PACK_BUS_BASE_OFFSET + pos * (head_size / PACK_SIZE / 2);
        int freq_cis_imag_pack_base_offset = FREQ_CIS_IMAG_PACK_BUS_BASE_OFFSET + pos * (head_size / PACK_SIZE / 2);
        int key_cache_pack_base_offset = KV_CACHE_PACK_BUS_BASE_OFFSET + i * (seq_len * kv_dim / PACK_SIZE_QUANTIZE / 4);
        int value_cache_pack_base_offset = KV_CACHE_PACK_BUS_BASE_OFFSET + i * (seq_len * kv_dim / PACK_SIZE_QUANTIZE / 4);
        int weight_bus_base_offset = i * STRIDE_WEIGHT_BUS;
        int scale_bus_base_offset = i * STRIDE_SCALE_BUS;
        int weight_q_pack_base_offset = weight_bus_base_offset + MATMUL_Q_GENERATION_WEIGHT_OFFSET;
        int weight_k_pack_base_offset = weight_bus_base_offset + MATMUL_K_GENERATION_WEIGHT_OFFSET;
        int weight_v_pack_base_offset = weight_bus_base_offset + MATMUL_V_GENERATION_WEIGHT_OFFSET;
        int weight_o_pack_base_offset = weight_bus_base_offset + MATMUL_O_GENERATION_WEIGHT_OFFSET;
        int weight_1_pack_base_offset = weight_bus_base_offset + MATMUL_1_GENERATION_WEIGHT_OFFSET;
        int weight_3_pack_base_offset = weight_bus_base_offset + MATMUL_3_GENERATION_WEIGHT_OFFSET;
        int weight_2_pack_base_offset = weight_bus_base_offset + MATMUL_2_GENERATION_WEIGHT_OFFSET;
        int weight_logits_pack_base_offset = weight_bus_base_offset + MATMUL_LOGITS_GENERATION_WEIGHT_OFFSET;
        int scale_q_pack_base_offset = scale_bus_base_offset + MATMUL_Q_GENERATION_SCALE_OFFSET;
        int scale_k_pack_base_offset = scale_bus_base_offset + MATMUL_K_GENERATION_SCALE_OFFSET;
        int scale_v_pack_base_offset = scale_bus_base_offset + MATMUL_V_GENERATION_SCALE_OFFSET;
        int scale_o_pack_base_offset = scale_bus_base_offset + MATMUL_O_GENERATION_SCALE_OFFSET;
        int scale_1_pack_base_offset = scale_bus_base_offset + MATMUL_1_GENERATION_SCALE_OFFSET;
        int scale_3_pack_base_offset = scale_bus_base_offset + MATMUL_3_GENERATION_SCALE_OFFSET;
        int scale_2_pack_base_offset = scale_bus_base_offset + MATMUL_2_GENERATION_SCALE_OFFSET;
        int scale_logits_pack_base_offset = weight_bus_base_offset + MATMUL_LOGITS_GENERATION_SCALE_OFFSET;

        MaxiOffset maxi_offset(
            weight_rmsnorm_att_pack_base_offset,
            weight_rmsnorm_ffn_pack_base_offset,
            WEIGHT_RMSNORM_FINAL_PACK_BASE_OFFSET,
            freq_cis_real_pack_base_offset,
            freq_cis_imag_pack_base_offset,
            key_cache_pack_base_offset,
            value_cache_pack_base_offset,
            weight_q_pack_base_offset,
            weight_k_pack_base_offset,
            weight_v_pack_base_offset,
            weight_o_pack_base_offset,
            weight_1_pack_base_offset,
            weight_3_pack_base_offset,
            weight_2_pack_base_offset,
            weight_logits_pack_base_offset,
            scale_q_pack_base_offset,
            scale_k_pack_base_offset,
            scale_v_pack_base_offset,
            scale_o_pack_base_offset,
            scale_1_pack_base_offset,
            scale_3_pack_base_offset,
            scale_2_pack_base_offset,
            scale_logits_pack_base_offset
        );

        decoder_top(
            pos,
            i,
            decoder_buffer,
            weight_port,
            maxi_offset
        );
    }
    save_decoder_buffer_x(decoder_buffer.logits, weight_port.ddr_bus_0, X_PACK_OUT_BASE_OFFSET);
}
