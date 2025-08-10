#include "../include/multiattention_top.h"

void softmax(DataType* output, DataType* input, int n) {
    #pragma HLS inline off
    DataType max_val = input[0];
    softmax_loop_max:
    for(int i = 1; i < n; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=seq_len max=seq_len
        max_val = (input[i] > max_val) ? input[i] : max_val;
    }

    DataType sum(0.0f);
    softmax_loop_sum:
    for(int i = 0; i < n; i++) {
        #pragma HLS PIPELINE II=3
        #pragma HLS LOOP_TRIPCOUNT min=seq_len max=seq_len
        DataType val = hls::exp(input[i] - max_val);
        output[i] = val;
        sum += val;
    }

    DataType inv_sum = (DataType)1.0f / sum;
    softmax_loop_inv_sum:
    for(int i = 0; i < n; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=seq_len max=seq_len
        output[i] = output[i] * inv_sum;
    }
}

void load_key_cache_buf(
    DataType key_cache_buf[MULTIATTENTION_DEPTH][head_size],
    DataType_quantize_pack_maxi weight_bus_1,
    DataType_quantize_pack_maxi weight_bus_2,
    DataType_quantize_pack_maxi weight_bus_3,
    DataType_quantize_pack_maxi weight_bus_4,
    int key_cache_pack_base_offset,
    int idx_row, int idx_col, int row_len
) {
    #pragma HLS inline
    const int base_offset = idx_row * row_len + idx_col + key_cache_pack_base_offset;
    int now = base_offset;
    DataType_pack pack[head_size*DATATYPE_WIDTH/QUANTIZE_WIDTH/PACK_SIZE_QUANTIZE];
    load_key_cache_buf_loop2:
    for(int i = 0; i < MULTIATTENTION_DEPTH; i++) {
        #pragma HLS UNROLL off=true
        #pragma HLS PIPELINE II=2

        weight_bus_1.read_request(now, 1);
        weight_bus_2.read_request(now+1, 1);
        weight_bus_3.read_request(now+2, 1);
        weight_bus_4.read_request(now+3, 1);
        weight_bus_1.read_request(now+4, 1);
        weight_bus_2.read_request(now+5, 1);
        weight_bus_3.read_request(now+6, 1);
        weight_bus_4.read_request(now+7, 1);
        pack[0] = weight_bus_1.read();
        pack[1] = weight_bus_2.read();
        pack[2] = weight_bus_3.read();
        pack[3] = weight_bus_4.read();
        pack[4] = weight_bus_1.read();
        pack[5] = weight_bus_2.read();
        pack[6] = weight_bus_3.read();
        pack[7] = weight_bus_4.read();
        now += row_len;
        for(int j = 0; j < head_size; j++) {
            #pragma HLS UNROLL factor=PACK_SIZE
            const int idx = j>>PACK_SIZE_LOG2;
            const int offset_key_cache = j & MASK_PACK;
            key_cache_buf[i][j] = from_bits(pack[idx](PKR(offset_key_cache)));
        }
    }
}

void load_q_buf(
    DataType q_buf[head_size],
    DataType* q
) {
    #pragma HLS inline
    load_q_buf_loop:
    for(int i = 0; i < head_size; ++ i) {
        #pragma HLS PIPELINE II=1
        q_buf[i] = q[i];
    }
}

void compute_score_tile(
    DataType score_buf[MULTIATTENTION_DEPTH],
    DataType q_buf[head_size],
    DataType key_cache_buf[MULTIATTENTION_DEPTH][head_size]
) {
    #pragma HLS inline
    DataType buffer[head_size<<1];
    #pragma HLS ARRAY_PARTITION type=complete dim=0 variable=buffer
    compute_score_tile_loop:
    for(int i = 0; i < MULTIATTENTION_DEPTH; i++) {
        #pragma HLS UNROLL off=true
        #pragma HLS PIPELINE II=1
        for(int j = 0; j < head_size; j++) {
            #pragma HLS UNROLL
            buffer[j] = q_buf[j] * key_cache_buf[i][j];
        }

        uint16_t cnt = head_size;
        uint16_t q = 0;
        for(int j = HEAD_SIZE_LOG2-1; j >= 0; j --) {
            #pragma HLS UNROLL
            for(int k = 0; k < (1<<j); k++) {
                #pragma HLS UNROLL
                buffer[cnt++] = buffer[q] + buffer[q+1];
                q += 2;
            }
        }
        score_buf[i] = buffer[cnt-1] / (DataType)std::sqrt(head_size * 1.0f);
    }
}

void save_score_buf(
    DataType* attention,
    DataType score_buf[MULTIATTENTION_DEPTH],
    int idx_row
) {
    #pragma HLS inline
    save_score_buf_loop:
    for(int i = 0; i < MULTIATTENTION_DEPTH; i++) {
        #pragma HLS UNROLL off=true
        #pragma HLS PIPELINE II=1
        attention[idx_row+i] = score_buf[i];
    }
}

void compute_score(
    DataType* attention,
    DataType* q,
    DataType_quantize_pack_maxi weight_bus_1,
    DataType_quantize_pack_maxi weight_bus_2,
    DataType_quantize_pack_maxi weight_bus_3,
    DataType_quantize_pack_maxi weight_bus_4,
    int key_cache_pack_base_offset,
    int pos,
    int head
) {
    #pragma HLS inline off
    DataType key_cache_buf[MULTIATTENTION_DEPTH][head_size];
    #pragma HLS ARRAY_PARTITION variable=key_cache_buf dim=2 complete
    DataType score_buf[MULTIATTENTION_DEPTH];
    DataType q_buf[head_size];
    #pragma HLS ARRAY_PARTITION variable=q_buf dim=1 complete

    const int loop_seq = (pos+1+MULTIATTENTION_DEPTH-1) >> MULTIATTENTION_DEPTH_LOG2;

    const int idx_col = head/kv_mul*(head_size*DATATYPE_WIDTH/QUANTIZE_WIDTH/PACK_SIZE_QUANTIZE);

    constexpr int row_len = (kv_dim*DATATYPE_WIDTH/QUANTIZE_WIDTH/PACK_SIZE_QUANTIZE);
    compute_score_loop:
    for(int i = 0; i < loop_seq; i++) {
        #pragma HLS PIPELINE off
        #pragma HLS LOOP_TRIPCOUNT min=seq_len/MULTIATTENTION_DEPTH max=seq_len/MULTIATTENTION_DEPTH

        load_q_buf(q_buf, q+head*head_size);
        load_key_cache_buf(key_cache_buf, weight_bus_1, weight_bus_2, weight_bus_3, weight_bus_4, key_cache_pack_base_offset, i*MULTIATTENTION_DEPTH, idx_col, row_len);
        compute_score_tile(score_buf, q_buf, key_cache_buf);
        save_score_buf(attention, score_buf, i*MULTIATTENTION_DEPTH);
    }
}

void load_value_cache_buf(
    DataType value_cache_buf[MULTIATTENTION_DEPTH][head_size],
    DataType_quantize_pack_maxi weight_bus_1,
    DataType_quantize_pack_maxi weight_bus_2,
    DataType_quantize_pack_maxi weight_bus_3,
    DataType_quantize_pack_maxi weight_bus_4,
    int value_cache_pack_base_offset,
    int idx_row, int idx_col, int row_len
) {
    #pragma HLS inline
    const int base_offset = idx_row * row_len + idx_col + value_cache_pack_base_offset;
    int now = base_offset;
    DataType_pack pack[head_size*DATATYPE_WIDTH/QUANTIZE_WIDTH/PACK_SIZE_QUANTIZE];
    load_value_cache_buf_loop2:
    for(int i = 0; i < MULTIATTENTION_DEPTH; i++) {
        #pragma HLS UNROLL off=true
        #pragma HLS PIPELINE II=2

        weight_bus_1.read_request(now, 1);
        weight_bus_2.read_request(now+1, 1);
        weight_bus_3.read_request(now+2, 1);
        weight_bus_4.read_request(now+3, 1);
        weight_bus_1.read_request(now+4, 1);
        weight_bus_2.read_request(now+5, 1);
        weight_bus_3.read_request(now+6, 1);
        weight_bus_4.read_request(now+7, 1);
        pack[0] = weight_bus_1.read();
        pack[1] = weight_bus_2.read();
        pack[2] = weight_bus_3.read();
        pack[3] = weight_bus_4.read();
        pack[4] = weight_bus_1.read();
        pack[5] = weight_bus_2.read();
        pack[6] = weight_bus_3.read();
        pack[7] = weight_bus_4.read();
        now += row_len;
        for(int j = 0; j < head_size; j++) {
            #pragma HLS UNROLL factor=PACK_SIZE
            const int idx = j>>PACK_SIZE_LOG2;
            const int offset_value_cache = j & MASK_PACK;
            value_cache_buf[i][j] = from_bits(pack[idx](PKR(offset_value_cache)));
        }
    }
}

void load_attention_buf(
    DataType attention_buf[MULTIATTENTION_DEPTH],
    DataType attention[seq_len],
    int idx_row
) {
    #pragma HLS inline
    load_attention_buf_loop:
    for(int i = 0; i < MULTIATTENTION_DEPTH; i++) {
        #pragma HLS UNROLL off=true
        #pragma HLS PIPELINE II=1
        attention_buf[i] = attention[idx_row+i];
    }
}

void compute_output_tile(
    DataType output_buf[head_size],
    DataType attention_buf[MULTIATTENTION_DEPTH],
    DataType value_cache_buf[MULTIATTENTION_DEPTH][head_size],
    int pos, int idx_row
) {
    #pragma HLS inline

    for(int i = 0; i < head_size; i++) {
        #pragma HLS UNROLL
        output_buf[i] = (DataType)0.0f;
    }

    compute_output_tile_loop:
    for(uint16_t i = 0; i < MULTIATTENTION_DEPTH; i++) {
        #pragma HLS UNROLL off=true
        #pragma HLS PIPELINE II=3
        ap_uint<13> idx_row_plus_i = (ap_uint<13>)idx_row + (ap_uint<13>)i;
        if ((ap_uint<13>)pos < idx_row_plus_i) break;
        for(int j = 0; j < head_size; j++) {
            #pragma HLS UNROLL
            output_buf[j] += attention_buf[i] * value_cache_buf[i][j];
        }
    }
}

void save_output_buf(
    DataType* output,
    DataType output_buf[head_size],
    int idx_col, bool is_first
) {
    #pragma HLS inline
    const int loop_end = head_size + idx_col;
    if(is_first) {
        save_output_buf_loop_first:
        for(int i = idx_col; i < loop_end; i++) {
            #pragma HLS PIPELINE II=1
            output[i] = output_buf[i-idx_col];
        }
    } else {
        save_output_buf_loop_not_first:
        for(int i = idx_col; i < loop_end; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS dependence variable=output inter false
            output[i] += output_buf[i-idx_col];
        }
    }

}

void compute_output(
    DataType* output,
    DataType attention[seq_len],
    DataType_quantize_pack_maxi weight_bus_1,
    DataType_quantize_pack_maxi weight_bus_2,
    DataType_quantize_pack_maxi weight_bus_3,
    DataType_quantize_pack_maxi weight_bus_4,
    int value_cache_pack_base_offset,
    int pos,
    int head
) {
    #pragma HLS inline off
    DataType value_cache_buf[MULTIATTENTION_DEPTH][head_size];
    #pragma HLS ARRAY_PARTITION variable=value_cache_buf dim=2 complete
    DataType output_buf[head_size];
    #pragma HLS ARRAY_PARTITION variable=output_buf dim=1 complete
    DataType attention_buf[MULTIATTENTION_DEPTH];

    const int loop_seq = (pos+1+MULTIATTENTION_DEPTH-1) >> MULTIATTENTION_DEPTH_LOG2;

    const int idx_col_value = head/kv_mul*(head_size*DATATYPE_WIDTH/QUANTIZE_WIDTH/PACK_SIZE_QUANTIZE);
    const int idx_col_output = head*head_size;

    constexpr int row_len = (kv_dim*DATATYPE_WIDTH/QUANTIZE_WIDTH/PACK_SIZE_QUANTIZE);

    compute_output_loop:
    for(int i = 0; i < loop_seq; i++) {
        #pragma HLS PIPELINE off
        #pragma HLS LOOP_TRIPCOUNT min=seq_len/MULTIATTENTION_DEPTH max=seq_len/MULTIATTENTION_DEPTH

        load_value_cache_buf(value_cache_buf, weight_bus_1, weight_bus_2, weight_bus_3, weight_bus_4, value_cache_pack_base_offset,i*MULTIATTENTION_DEPTH, idx_col_value, row_len);
        load_attention_buf(attention_buf, attention, i*MULTIATTENTION_DEPTH);
        compute_output_tile(output_buf, attention_buf, value_cache_buf, pos, i*MULTIATTENTION_DEPTH);
        save_output_buf(output, output_buf, idx_col_output, i == 0);
    }

}

void save_kv_cache(
    DataType_quantize_pack_maxi weight_bus_1,
    DataType_quantize_pack_maxi weight_bus_2,
    DataType_quantize_pack_maxi weight_bus_3,
    DataType_quantize_pack_maxi weight_bus_4,
    DataType_quantize_pack_maxi weight_bus_5,
    DataType_quantize_pack_maxi weight_bus_6,
    DataType_quantize_pack_maxi weight_bus_7,
    DataType_quantize_pack_maxi weight_bus_8,
    int key_cache_pack_base_offset,
    int value_cache_pack_base_offset,
    DataType* k,
    DataType* v,
    int pos
) {
    #pragma HLS inline
    const int offset = pos*(kv_dim/PACK_SIZE_QUANTIZE/4) + key_cache_pack_base_offset;
    const int req_size = kv_dim*DATATYPE_WIDTH/QUANTIZE_WIDTH/PACK_SIZE_QUANTIZE/4/16;
    save_kv_cache_loop1:
    for(int i = 0; i < req_size; i ++) {
        #pragma HLS PIPELINE II=1
        weight_bus_1.write_request(offset+i*16, 16);
        weight_bus_2.write_request(offset+i*16, 16);
        weight_bus_3.write_request(offset+i*16, 16);
        weight_bus_4.write_request(offset+i*16, 16);
        weight_bus_5.write_request(offset+i*16, 16);
        weight_bus_6.write_request(offset+i*16, 16);
        weight_bus_7.write_request(offset+i*16, 16);
        weight_bus_8.write_request(offset+i*16, 16);
    }
    DataType_quantize_pack key_cache_tmp;
    DataType_quantize_pack value_cache_tmp;
    int cnt= 0;
    save_kv_cache_loop2:
    for(int i = 0; i < dim; i ++) {
		#pragma HLS PIPELINE II=1
        const int offset = i & MASK_PACK;
        key_cache_tmp(PKR(offset)) = to_bits(k[i]);
        value_cache_tmp(PKR(offset)) = to_bits(v[i]);
        if(i % PACK_SIZE == PACK_SIZE - 1) {
            if(cnt % 4 == 0) weight_bus_1.write(key_cache_tmp), weight_bus_5.write(value_cache_tmp);
            else if(cnt % 4 == 1) weight_bus_2.write(key_cache_tmp), weight_bus_6.write(value_cache_tmp);
            else if(cnt % 4 == 2) weight_bus_3.write(key_cache_tmp), weight_bus_7.write(value_cache_tmp);
            else if(cnt % 4 == 3) weight_bus_4.write(key_cache_tmp), weight_bus_8.write(value_cache_tmp);
            cnt ++;
        }
    }

    save_kv_cache_loop3:
    for(int i = 0; i < req_size; i ++) {
        #pragma HLS PIPELINE II=1
        weight_bus_1.write_response();
        weight_bus_2.write_response();
        weight_bus_3.write_response();
        weight_bus_4.write_response();
        weight_bus_5.write_response();
        weight_bus_6.write_response();
        weight_bus_7.write_response();
        weight_bus_8.write_response();
    }
}

void multiattention(
    DataType* output,
    DataType_quantize_pack_maxi weight_bus_1,
    DataType_quantize_pack_maxi weight_bus_2,
    DataType_quantize_pack_maxi weight_bus_3,
    DataType_quantize_pack_maxi weight_bus_4,
    DataType_quantize_pack_maxi weight_bus_5,
    DataType_quantize_pack_maxi weight_bus_6,
    DataType_quantize_pack_maxi weight_bus_7,
    DataType_quantize_pack_maxi weight_bus_8,
    int key_cache_pack_base_offset,
    int value_cache_pack_base_offset,
    DataType* q,
    int pos
){
    #pragma HLS inline off

    DataType attention_in[seq_len];
    DataType attention_out[seq_len];

    multiattention_loop:
    for(int head = 0; head < n_heads; ++ head) {
        #pragma HLS UNROLL off=true
        #pragma HLS DATAFLOW
        compute_score(attention_in, q, weight_bus_1, weight_bus_2, weight_bus_3, weight_bus_4, key_cache_pack_base_offset, pos, head);
        softmax(attention_out, attention_in, pos+1);
        compute_output(output, attention_out, weight_bus_5, weight_bus_6, weight_bus_7, weight_bus_8, value_cache_pack_base_offset, pos, head);
    }
}

extern "C" void multiattention_top(
    DecoderBuffer& decoder_buffer,
    WeightPort& weight_port,
    MaxiOffset& maxi_offset,
    int pos,
    Controller* controller
){
    #pragma HLS inline off
    if(controller->run_multiattention == RUN) {
        save_kv_cache(weight_port.weight_bus_1, weight_port.weight_bus_2, weight_port.weight_bus_3, weight_port.weight_bus_4, weight_port.weight_bus_5, weight_port.weight_bus_6, weight_port.weight_bus_7, weight_port.weight_bus_8, maxi_offset.key_cache_pack_base_offset, maxi_offset.value_cache_pack_base_offset, decoder_buffer.k2, decoder_buffer.v, pos);
        multiattention(decoder_buffer.xb, weight_port.weight_bus_1, weight_port.weight_bus_2, weight_port.weight_bus_3, weight_port.weight_bus_4, weight_port.weight_bus_5, weight_port.weight_bus_6, weight_port.weight_bus_7, weight_port.weight_bus_8, maxi_offset.key_cache_pack_base_offset, maxi_offset.value_cache_pack_base_offset, decoder_buffer.q2, pos);
        controller->index_quantize = QUANTIZE_MULTIATTENTION;
        controller->run_quantize = RUN;
        controller->run_multiattention = LAZY;
    }
}
