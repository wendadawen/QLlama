#include "../include/nonlinear_top.h"

void load_rmsnorm_weight_buf( DataType weight_buf[dim], DataType_pack_maxi weight, int weight_pack_base_offset) {
    #pragma HLS inline off

    const int pack_len = dim/PACK_SIZE;

    const int req_size = pack_len / 64;
    load_rmsnorm_weight_buf_loop1:
    for(int i = 0; i < req_size; i ++) {
        #pragma HLS PIPELINE II=1
        weight.read_request(i*64 + weight_pack_base_offset, 64);
    }

    DataType_pack pack;
    load_rmsnorm_weight_buf_loop2:
    for(int i = 0; i < dim; i++) {
        #pragma HLS PIPELINE II=1
        if(i % PACK_SIZE == 0) pack = weight.read();
        const int offset = i&MASK_PACK;
        weight_buf[i] = from_bits(pack(PKR(offset)));
    }
}

void rmsnorm_calc_rms(DataType* input, DataType& ss) {
    #pragma HLS inline off
    DataType s[4];
    for(int i = 0; i < 4; i++) {
        #pragma HLS UNROLL off=true
        s[i] = 0.0f;
    }
    #pragma HLS ARRAY_PARTITION variable=s dim=1 complete
    rmsnorm_calc_rms_loop:
    for(int i = 0; i < dim; i+=4) {
        #pragma HLS PIPELINE II=4

        for(int j = 0; j < 4; j ++) {
            #pragma HLS UNROLL off=true
            DataType val1 = input[i+j];
            s[j] += val1 * val1;
        }
    }
    DataType s1 = s[0] + s[1];
    DataType s2 = s[2] + s[3];
    ss = s1 + s2;
    ss = ss / (DataType)dim;
    ss = hls::rsqrt(ss + (DataType)1e-5f);
}

void rmsnorm_normalize(DataType* output, DataType* input, DataType ss, DataType* weight_buf) {
    #pragma HLS inline off
    rmsnorm_normalize_loop:
    for(int i = 0; i < dim; i++) {
        #pragma HLS PIPELINE II=1
        output[i] = input[i] * ss * weight_buf[i];
    }
}

void rmsnorm(
    DataType* output,
    DataType* input,
    DataType_pack_maxi weight,
    int weight_pack_base_offset
) {
    #pragma HLS inline off
    #pragma HLS DATAFLOW
    DataType weight_buf[dim];
    #pragma HLS bind_storage variable=weight_buf type=RAM_2P impl=uram
    DataType ss(0.0f);

    load_rmsnorm_weight_buf(weight_buf, weight, weight_pack_base_offset);
    rmsnorm_calc_rms(input, ss);
    rmsnorm_normalize(output, input, ss, weight_buf);
}

void load_rope_freq_cis(
    DataType freq_cis_real_buf[head_size>>1],
    DataType freq_cis_imag_buf[head_size>>1],
    DataType_pack_maxi freq_cis_real,
    DataType_pack_maxi freq_cis_imag,
    int freq_cis_real_pack_base_offset,
    int freq_cis_imag_pack_base_offset
) {
    #pragma HLS inline

    const int loop_head_size = head_size >> 1;
    const int pack_len = loop_head_size/PACK_SIZE;

    freq_cis_real.read_request(freq_cis_real_pack_base_offset, pack_len);
    freq_cis_imag.read_request(freq_cis_imag_pack_base_offset, pack_len);

    DataType_pack pack_real;
    DataType_pack pack_imag;
    load_rope_freq_cis_loop1:
    for(int i = 0; i < (loop_head_size >> PACK_SIZE_LOG2); i++) {
        #pragma HLS LOOP_FLATTEN off
        #pragma HLS PIPELINE off
        #pragma HLS UNROLL off=true
        pack_real = freq_cis_real.read();
        pack_imag = freq_cis_imag.read();
        load_rope_freq_cis_loop2:
        for(int j = 0; j < PACK_SIZE; j++) {
            #pragma HLS PIPELINE II=1
            const int offset = (i<<PACK_SIZE_LOG2) + j;
            freq_cis_real_buf[offset] = from_bits(pack_real(PKR(j)));
            freq_cis_imag_buf[offset] = from_bits(pack_imag(PKR(j)));
        }
    }
}

void rope(
    DataType* q_out,
    DataType* k_out,
    DataType* q_in,
    DataType* k_in,
    DataType_pack_maxi freq_cis_real,
    DataType_pack_maxi freq_cis_imag,
    int freq_cis_real_pack_base_offset,
    int freq_cis_imag_pack_base_offset
) {
    #pragma HLS inline off
    DataType freq_cis_real_buf[head_size>>1];
    DataType freq_cis_imag_buf[head_size>>1];
    load_rope_freq_cis(freq_cis_real_buf, freq_cis_imag_buf, freq_cis_real, freq_cis_imag, freq_cis_real_pack_base_offset, freq_cis_imag_pack_base_offset);

    rope_loop1:
    for (int head = 0; head < n_heads; ++head) {
        #pragma HLS UNROLL off=true
        int start = head * head_size;
        rope_loop2:
        for (int i = 0; i < head_size; i += 2) {
            #pragma HLS UNROLL off=true
            #pragma HLS PIPELINE II=1
            DataType fcr = freq_cis_real_buf[i >> 1];
            DataType fci = freq_cis_imag_buf[i >> 1];
            DataType q0 = *((DataType*)q_in);q_in++;
            DataType q1 = *((DataType*)q_in);q_in++;
            q_out[start + i]     = q0 * fcr - q1 * fci;
            q_out[start + i + 1] = q0 * fci + q1 * fcr;
            if(start + i < kv_dim) {
                DataType k0 = *((DataType*)k_in);k_in++;
                DataType k1 = *((DataType*)k_in);k_in++;
                k_out[start + i]     = k0 * fcr - k1 * fci;
                k_out[start + i + 1] = k0 * fci + k1 * fcr;
            }
        }
    }
}

void swiglu(
    DataType* output,
    DataType* input,
    DataType* input2
) {
    #pragma HLS inline off

    swiglu_loop:
    for(int i = 0; i < hidden_dim; i++) {
        #pragma HLS PIPELINE II=1
        DataType x = input[i];
        DataType sigmoid = (DataType)1.0f / ((DataType)1.0f + hls::exp(-x));
        output[i] = x * sigmoid * input2[i];
    }
}

extern "C" void nonlinear_top(
    DecoderBuffer& decoder_buffer,
    WeightPort& weight_port,
    MaxiOffset& maxi_offset,
    Controller* controller
) {
    #pragma HLS inline off
    if(controller->run_rmsnorm == RUN) {
        if(controller->index_rmsnorm == RMSNORM_ATT) {
            rmsnorm(decoder_buffer.xb, decoder_buffer.x, weight_port.ddr_bus_0, maxi_offset.weight_rmsnorm_att_pack_base_offset);
            controller->index_quantize = QUANTIZE_RMSNORM_ATT;
        }
        else if(controller->index_rmsnorm == RMSNORM_FFN) {
            rmsnorm(decoder_buffer.xb, decoder_buffer.x, weight_port.ddr_bus_0, maxi_offset.weight_rmsnorm_ffn_pack_base_offset);
            controller->index_quantize = QUANTIZE_RMSNORM_FFN;
        }
        else if(controller->index_rmsnorm == RMSNORM_FINAL) {
            rmsnorm(decoder_buffer.xb, decoder_buffer.x, weight_port.ddr_bus_0, maxi_offset.weight_rmsnorm_final_pack_base_offset);
            controller->index_quantize = QUANTIZE_RMSNORM_FINAL;
        }
        controller->run_quantize = RUN;
        controller->run_rmsnorm = LAZY;
    }
    else if(controller->run_swiglu == RUN) {
        swiglu(decoder_buffer.hb3, decoder_buffer.hb, decoder_buffer.hb2);
        controller->index_quantize = QUANTIZE_SWIGLU;
        controller->run_quantize = RUN;
        controller->run_swiglu = LAZY;
    }
    else if(controller->run_rope == RUN) {
        rope(decoder_buffer.q2, decoder_buffer.k2, decoder_buffer.q, decoder_buffer.k, weight_port.ddr_bus_0, weight_port.ddr_bus_0, maxi_offset.freq_cis_real_pack_base_offset, maxi_offset.freq_cis_imag_pack_base_offset);
        controller->run_rope = LAZY;
        controller->run_multiattention = RUN;
    }
}
