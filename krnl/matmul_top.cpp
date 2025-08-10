#include "../include/matmul_top.h"

void load_weight_buf(
    DataType_quantize weight_buf[MATMUL_DEPTH][MATMUL_WIDTH],
    DataType_quantize_pack_maxi weight_bus1,
    DataType_quantize_pack_maxi weight_bus2,
    DataType_quantize_pack_maxi weight_bus3,
    DataType_quantize_pack_maxi weight_bus4,
    DataType_quantize_pack_maxi weight_bus5,
    DataType_quantize_pack_maxi weight_bus6,
    DataType_quantize_pack_maxi weight_bus7,
    DataType_quantize_pack_maxi weight_bus8,
    DataType_quantize_pack_maxi weight_bus9,
    DataType_quantize_pack_maxi weight_bus10,
    DataType_quantize_pack_maxi weight_bus11,
    DataType_quantize_pack_maxi weight_bus12,
    DataType_quantize_pack_maxi weight_bus13,
    DataType_quantize_pack_maxi weight_bus14,
    DataType_quantize_pack_maxi weight_bus15,
    DataType_quantize_pack_maxi weight_bus16,
    int base_offset_bus,
    int idx_row, int idx_col, int row_len, int n, int m
) {
    #pragma HLS inline off
    const int base_offset = base_offset_bus + (idx_row * row_len + idx_col) / PC;
    DataType_quantize_pack pack[PC];
    #pragma HLS ARRAY_PARTITION variable=pack dim=1 complete
    const int stride = row_len/PC;
    int now = 0;
    load_weight_buf_loop2:
    for(int i = 0; i < MATMUL_DEPTH; i++) {
        #pragma HLS PIPELINE II=1

        const int offset = base_offset + now; now += stride;
        if((uint16_t)i >= (uint16_t)n) break;
        weight_bus1.read_request(offset, 1);
        weight_bus2.read_request(offset, 1);
        weight_bus3.read_request(offset, 1);
        weight_bus4.read_request(offset, 1);
        weight_bus5.read_request(offset, 1);
        weight_bus6.read_request(offset, 1);
        weight_bus7.read_request(offset, 1);
        weight_bus8.read_request(offset, 1);
        weight_bus9.read_request(offset, 1);
        weight_bus10.read_request(offset, 1);
        weight_bus11.read_request(offset, 1);
        weight_bus12.read_request(offset, 1);
        weight_bus13.read_request(offset, 1);
        weight_bus14.read_request(offset, 1);
        weight_bus15.read_request(offset, 1);
        weight_bus16.read_request(offset, 1);
        pack[0] = weight_bus1.read();
        pack[1] = weight_bus2.read();
        pack[2] = weight_bus3.read();
        pack[3] = weight_bus4.read();
        pack[4] = weight_bus5.read();
        pack[5] = weight_bus6.read();
        pack[6] = weight_bus7.read();
        pack[7] = weight_bus8.read();
        pack[8] = weight_bus9.read();
        pack[9] = weight_bus10.read();
        pack[10] = weight_bus11.read();
        pack[11] = weight_bus12.read();
        pack[12] = weight_bus13.read();
        pack[13] = weight_bus14.read();
        pack[14] = weight_bus15.read();
        pack[15] = weight_bus16.read();
        for(int j = 0; j < MATMUL_WIDTH; j++) {
            #pragma HLS UNROLL
            const int t = j>>PACK_SIZE_QUANTIZE_LOG2;
            const int offset_weight = j & MASK_PACK_QUANTIZE;
            weight_buf[i][j] = from_bits_quantize(pack[t](PKR_Q(offset_weight)));
        }
    }
}

void load_weight_scale_buf(
    DataType_scale weight_scale_buf[MATMUL_DEPTH][MATMUL_WIDTH >> GS_LOG2],
    DataType_scale_pack_maxi scale_bus,
    int scale_base_offset,
    int idx_row, int idx_col, int row_len, int n, int m
) {
    #pragma HLS inline off
    const int base_offset = scale_base_offset + idx_row * row_len + idx_col;
    const int WIDTH_div_GS = MATMUL_WIDTH >> GS_LOG2;
    int now = 0;
    DataType_scale_pack pack;
    #pragma HLS ARRAY_PARTITION variable=pack dim=1 complete
    load_weight_scale_buf_loop2:
    for(int i = 0; i < MATMUL_DEPTH; i++) {
        #pragma HLS PIPELINE II=1

        const int offset = base_offset + now; now += row_len;
        if((uint16_t)i >= (uint16_t)n) break;
        scale_bus.read_request(offset, 1);
        pack = scale_bus.read();
        for(int j = 0; j < WIDTH_div_GS; j++) {
            #pragma HLS UNROLL
            const int offset_scale = j & MASK_PACK_SCALE;
            weight_scale_buf[i][j] = from_bits_scale(pack(PKR_S(offset_scale)));
        }
    }
}

void load_input_buf(
    DataType_quantize input_buf[MATMUL_WIDTH],
    DataType_quantize* input,
    int idx_col, int m
) {
    #pragma HLS inline off
    load_input_buf_loop:
    for(int i = 0; i < MATMUL_WIDTH; i+=2) {
        #pragma HLS PIPELINE II=1
        input_buf[i] = input[idx_col+i];
        input_buf[i+1] = input[idx_col+i+1];
    }
}

void load_input_scale_buf(
    DataType_scale input_scale_buf[MATMUL_WIDTH >> GS_LOG2],
    DataType_scale* input_scale,
    int idx_col,int m
) {
    #pragma HLS inline off
    const int WIDTH_div_GS = MATMUL_WIDTH >> GS_LOG2;
    load_input_scale_buf_loop:
    for(int i = 0; i < WIDTH_div_GS; i++) {
        #pragma HLS PIPELINE II=1
        input_scale_buf[i] = input_scale[idx_col+i];
    }
}

void save_output_buf(
    DataType* output,
    DataType output_buf[MATMUL_DEPTH],
    int idx_row, int first_col, int n
) {
    #pragma HLS inline off
    const int loop_n = (n < MATMUL_DEPTH) ? n+idx_row : MATMUL_DEPTH+idx_row;
    if (first_col) {

        save_output_buf_loop_assign:
        for(int i = idx_row; i < loop_n; i++) {
            #pragma HLS PIPELINE II=1
            output[i] = output_buf[i-idx_row];
        }
    } else {

        save_output_buf_loop_accumulate:
        for(int i = idx_row; i < loop_n; i++) {
            #pragma HLS PIPELINE II=1
            output[i] += output_buf[i-idx_row];
        }
    }
}

void matmul_tile(
    DataType output_buf[MATMUL_DEPTH],
    DataType_quantize input_buf[MATMUL_WIDTH],
    DataType_scale input_scale_buf[MATMUL_WIDTH >> GS_LOG2],
    DataType_quantize weight_buf[MATMUL_DEPTH][MATMUL_WIDTH],
    DataType_scale weight_scale_buf[MATMUL_DEPTH][MATMUL_WIDTH >> GS_LOG2],
    int n,
    int m
){
    #pragma HLS inline off
    const int WIDTH_div_GS = MATMUL_WIDTH >> GS_LOG2;
    DataType output_buf_i[WIDTH_div_GS];
    #pragma HLS ARRAY_PARTITION variable=output_buf_i dim=1 complete
    DataType output_buf_i_sum[WIDTH_div_GS];
    #pragma HLS ARRAY_PARTITION variable=output_buf_i_sum dim=1 complete
    matmul_tile_loop:
    for(int i = 0; i < MATMUL_DEPTH; i++) {
        #pragma HLS PIPELINE II=1
        if(i >= n) break;
        for(int j = 0; j < WIDTH_div_GS; j++) {
            #pragma HLS UNROLL
            ap_fixed<32, 6> acc = 0;
            const int offset = j<<GS_LOG2;
            for(int k = 0; k < GS; k++) {
                #pragma HLS UNROLL
                acc += input_buf[offset+k] * weight_buf[i][offset+k];
            }

            int scale = weight_scale_buf[i][j] + input_scale_buf[j] - 254;
            DataType temp = (DataType) (std::ldexp((DataType)acc, scale));
            if((j << GS_LOG2) >= m) output_buf_i[j] = 0;
            else output_buf_i[j] = temp;
        }
        output_buf_i_sum[0] = output_buf_i[0] + output_buf_i[1];
        output_buf_i_sum[1] = output_buf_i[2] + output_buf_i[3];
        output_buf_i_sum[2] = output_buf_i[4] + output_buf_i[5];
        output_buf_i_sum[3] = output_buf_i[6] + output_buf_i[7];
        output_buf_i_sum[4] = output_buf_i[8] + output_buf_i[9];
        output_buf_i_sum[5] = output_buf_i[10] + output_buf_i[11];
        output_buf_i_sum[6] = output_buf_i[12] + output_buf_i[13];
        output_buf_i_sum[7] = output_buf_i[14] + output_buf_i[15];
        output_buf_i_sum[8] = output_buf_i[16] + output_buf_i[17];
        output_buf_i_sum[9] = output_buf_i[18] + output_buf_i[19];
        output_buf_i_sum[10] = output_buf_i[20] + output_buf_i[21];
        output_buf_i_sum[11] = output_buf_i[22] + output_buf_i[23];
        output_buf_i_sum[12] = output_buf_i[24] + output_buf_i[25];
        output_buf_i_sum[13] = output_buf_i[26] + output_buf_i[27];
        output_buf_i_sum[14] = output_buf_i[28] + output_buf_i[29];
        output_buf_i_sum[15] = output_buf_i[30] + output_buf_i[31];

        output_buf_i_sum[16] = output_buf_i_sum[0] + output_buf_i_sum[1];
        output_buf_i_sum[17] = output_buf_i_sum[2] + output_buf_i_sum[3];
        output_buf_i_sum[18] = output_buf_i_sum[4] + output_buf_i_sum[5];
        output_buf_i_sum[19] = output_buf_i_sum[6] + output_buf_i_sum[7];
        output_buf_i_sum[20] = output_buf_i_sum[8] + output_buf_i_sum[9];
        output_buf_i_sum[21] = output_buf_i_sum[10] + output_buf_i_sum[11];
        output_buf_i_sum[22] = output_buf_i_sum[12] + output_buf_i_sum[13];
        output_buf_i_sum[23] = output_buf_i_sum[14] + output_buf_i_sum[15];

        output_buf_i_sum[24] = output_buf_i_sum[16] + output_buf_i_sum[17];
        output_buf_i_sum[25] = output_buf_i_sum[18] + output_buf_i_sum[19];
        output_buf_i_sum[26] = output_buf_i_sum[20] + output_buf_i_sum[21];
        output_buf_i_sum[27] = output_buf_i_sum[22] + output_buf_i_sum[23];

        output_buf_i_sum[28] = output_buf_i_sum[24] + output_buf_i_sum[25];
        output_buf_i_sum[29] = output_buf_i_sum[26] + output_buf_i_sum[27];

        output_buf_i_sum[30] = output_buf_i_sum[28] + output_buf_i_sum[29];

        output_buf[i] = output_buf_i_sum[30];
    }
}

void matmul(
    DataType* output,
    DataType_quantize* input,
    DataType_scale* input_scale,
    DataType_quantize_pack_maxi weight_bus1,
    DataType_quantize_pack_maxi weight_bus2,
    DataType_quantize_pack_maxi weight_bus3,
    DataType_quantize_pack_maxi weight_bus4,
    DataType_quantize_pack_maxi weight_bus5,
    DataType_quantize_pack_maxi weight_bus6,
    DataType_quantize_pack_maxi weight_bus7,
    DataType_quantize_pack_maxi weight_bus8,
    DataType_quantize_pack_maxi weight_bus9,
    DataType_quantize_pack_maxi weight_bus10,
    DataType_quantize_pack_maxi weight_bus11,
    DataType_quantize_pack_maxi weight_bus12,
    DataType_quantize_pack_maxi weight_bus13,
    DataType_quantize_pack_maxi weight_bus14,
    DataType_quantize_pack_maxi weight_bus15,
    DataType_quantize_pack_maxi weight_bus16,
    int weight_base_offset,
    DataType_scale_pack_maxi scale_bus,
    int scale_base_offset,
    int n,
    int m
) {
    #pragma HLS inline off

    const int loop_n = (n + MATMUL_DEPTH - 1) >> MATMUL_DEPTH_LOG2;
    const int loop_m = m >> MATMUL_WIDTH_LOG2;
    const int row_len_weight = m >> PACK_SIZE_QUANTIZE_LOG2;
    const int row_len_scale = (m >> GS_LOG2) >> PACK_SIZE_SCALE_LOG2;
    const int WIDTH_div_PACK = MATMUL_WIDTH >> PACK_SIZE_QUANTIZE_LOG2;
    const int WIDTH_div_GS = MATMUL_WIDTH >> GS_LOG2;
    const int WIDTH_div_GS_PACK = WIDTH_div_GS >> PACK_SIZE_SCALE_LOG2;

    DataType_quantize input_buf_0[MATMUL_WIDTH];
    DataType_quantize input_buf_1[MATMUL_WIDTH];
    DataType_quantize weight_buf_0[MATMUL_DEPTH][MATMUL_WIDTH];
    DataType_quantize weight_buf_1[MATMUL_DEPTH][MATMUL_WIDTH];
    DataType_scale weight_scale_buf_0[MATMUL_DEPTH][WIDTH_div_GS];
    DataType_scale weight_scale_buf_1[MATMUL_DEPTH][WIDTH_div_GS];
    #pragma HLS BIND_STORAGE variable=weight_scale_buf_0 type=ram_2p impl=uram
    #pragma HLS BIND_STORAGE variable=weight_scale_buf_1 type=ram_2p impl=uram
    DataType_scale input_scale_buf_0[WIDTH_div_GS];
    DataType_scale input_scale_buf_1[WIDTH_div_GS];
    DataType output_buf_0[MATMUL_DEPTH];
    DataType output_buf_1[MATMUL_DEPTH];
    #pragma HLS ARRAY_PARTITION variable=input_buf_0 dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=input_buf_1 dim=1 complete
    #pragma HLS ARRAY_RESHAPE variable=weight_buf_0 dim=2 complete
    #pragma HLS ARRAY_RESHAPE variable=weight_buf_1 dim=2 complete
    #pragma HLS ARRAY_RESHAPE variable=weight_scale_buf_0 dim=2 complete
    #pragma HLS ARRAY_RESHAPE variable=weight_scale_buf_1 dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=input_scale_buf_0 dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=input_scale_buf_1 dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=output_buf_0 dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=output_buf_1 dim=1 complete

    int pingpong = 0;
    int pre_calc_n = MATMUL_DEPTH;
    int post_calc_n = MATMUL_DEPTH;
    int now_calc_n = MATMUL_DEPTH;
    int pre_calc_m = MATMUL_WIDTH;
    load_weight_buf(weight_buf_0, weight_bus1, weight_bus2, weight_bus3, weight_bus4, weight_bus5, weight_bus6, weight_bus7, weight_bus8, weight_bus9, weight_bus10, weight_bus11, weight_bus12, weight_bus13, weight_bus14, weight_bus15, weight_bus16, weight_base_offset, 0, 0, row_len_weight, pre_calc_n, pre_calc_m);
    load_weight_scale_buf(weight_scale_buf_0, scale_bus, scale_base_offset, 0, 0, row_len_scale, pre_calc_n, pre_calc_m);
    load_input_scale_buf(input_scale_buf_0, input_scale, 0, pre_calc_m);
    load_input_buf(input_buf_0, input, 0, pre_calc_m);
    matmul_loop1:
    for(int i = 0 ; i < loop_m; i ++) {
        #pragma HLS PIPELINE off
        #pragma HLS LOOP_TRIPCOUNT min=hidden_dim/MATMUL_WIDTH max=hidden_dim/MATMUL_WIDTH
        matmul_loop2:
        for(int j = 0; j < loop_n; j++) {
            #pragma HLS PIPELINE off
            #pragma HLS LOOP_TRIPCOUNT min=hidden_dim/MATMUL_DEPTH max=hidden_dim/MATMUL_DEPTH

            int ii = i;
            int jj = j+1;
            if(i != loop_m-1 && j == loop_n-1) ii++, jj=0;

            int iii = i;
            int jjj = j-1;
            if(i != 0 && j == 0) iii--, jjj=loop_n-1;

            post_calc_n = now_calc_n;
            now_calc_n = pre_calc_n;
            if(n % MATMUL_DEPTH == 0) pre_calc_n = MATMUL_DEPTH;
            else pre_calc_n = jj == loop_n-1 ? n % MATMUL_DEPTH : MATMUL_DEPTH;
            if(m % MATMUL_WIDTH == 0) pre_calc_m = MATMUL_WIDTH;
            else pre_calc_m = ii == loop_m-1 ? m % MATMUL_WIDTH : MATMUL_WIDTH;

            if(pingpong == 0){

                if(i == 0 && j == 0){
                    load_weight_buf(weight_buf_1, weight_bus1, weight_bus2, weight_bus3, weight_bus4, weight_bus5, weight_bus6, weight_bus7, weight_bus8, weight_bus9, weight_bus10, weight_bus11, weight_bus12, weight_bus13, weight_bus14, weight_bus15, weight_bus16, weight_base_offset, jj<<MATMUL_DEPTH_LOG2, ii*WIDTH_div_PACK, row_len_weight, pre_calc_n, pre_calc_m);
                    matmul_tile(output_buf_0, input_buf_0, input_scale_buf_0, weight_buf_0, weight_scale_buf_0, now_calc_n, pre_calc_m);
                    load_weight_scale_buf(weight_scale_buf_1, scale_bus, scale_base_offset, jj<<MATMUL_DEPTH_LOG2, ii*WIDTH_div_GS_PACK, row_len_scale, pre_calc_n, pre_calc_m);
                    load_input_scale_buf(input_scale_buf_1, input_scale, ii*WIDTH_div_GS, pre_calc_m);
                    load_input_buf(input_buf_1, input, ii<<MATMUL_WIDTH_LOG2, pre_calc_m);
                } else if(i == loop_m-1 && j == loop_n-1){
                    matmul_tile(output_buf_0, input_buf_0, input_scale_buf_0, weight_buf_0, weight_scale_buf_0, now_calc_n, pre_calc_m);
                    save_output_buf(output, output_buf_1, jjj<<MATMUL_DEPTH_LOG2, iii==0, post_calc_n);
                } else {
                    load_weight_buf(weight_buf_1, weight_bus1, weight_bus2, weight_bus3, weight_bus4, weight_bus5, weight_bus6, weight_bus7, weight_bus8, weight_bus9, weight_bus10, weight_bus11, weight_bus12, weight_bus13, weight_bus14, weight_bus15, weight_bus16, weight_base_offset, jj<<MATMUL_DEPTH_LOG2, ii*WIDTH_div_PACK, row_len_weight, pre_calc_n, pre_calc_m);
                    matmul_tile(output_buf_0, input_buf_0, input_scale_buf_0, weight_buf_0, weight_scale_buf_0, now_calc_n, pre_calc_m);
                    load_weight_scale_buf(weight_scale_buf_1, scale_bus, scale_base_offset, jj<<MATMUL_DEPTH_LOG2, ii*WIDTH_div_GS_PACK, row_len_scale, pre_calc_n, pre_calc_m);
                    load_input_scale_buf(input_scale_buf_1, input_scale, ii*WIDTH_div_GS, pre_calc_m);
                    load_input_buf(input_buf_1, input, ii<<MATMUL_WIDTH_LOG2, pre_calc_m);
                    save_output_buf(output, output_buf_1, jjj<<MATMUL_DEPTH_LOG2, iii==0, post_calc_n);
                }
                pingpong = 1;

            } else {

                if(i == 0 && j == 0){
                    load_weight_buf(weight_buf_0, weight_bus1, weight_bus2, weight_bus3, weight_bus4, weight_bus5, weight_bus6, weight_bus7, weight_bus8, weight_bus9, weight_bus10, weight_bus11, weight_bus12, weight_bus13, weight_bus14, weight_bus15, weight_bus16, weight_base_offset, jj<<MATMUL_DEPTH_LOG2, ii*WIDTH_div_PACK, row_len_weight, pre_calc_n, pre_calc_m);
                    matmul_tile(output_buf_1, input_buf_1, input_scale_buf_1, weight_buf_1, weight_scale_buf_1, now_calc_n, pre_calc_m);
                    load_weight_scale_buf(weight_scale_buf_0, scale_bus, scale_base_offset, jj<<MATMUL_DEPTH_LOG2, ii*WIDTH_div_GS_PACK, row_len_scale, pre_calc_n, pre_calc_m);
                    load_input_scale_buf(input_scale_buf_0, input_scale, ii*WIDTH_div_GS, pre_calc_m);
                    load_input_buf(input_buf_0, input, ii<<MATMUL_WIDTH_LOG2, pre_calc_m);
                } else if(i == loop_m-1 && j == loop_n-1){
                    matmul_tile(output_buf_1, input_buf_1, input_scale_buf_1, weight_buf_1, weight_scale_buf_1, now_calc_n, pre_calc_m);
                    save_output_buf(output, output_buf_0, jjj<<MATMUL_DEPTH_LOG2, iii==0, post_calc_n);
                } else {
                    load_weight_buf(weight_buf_0, weight_bus1, weight_bus2, weight_bus3, weight_bus4, weight_bus5, weight_bus6, weight_bus7, weight_bus8, weight_bus9, weight_bus10, weight_bus11, weight_bus12, weight_bus13, weight_bus14, weight_bus15, weight_bus16, weight_base_offset, jj<<MATMUL_DEPTH_LOG2, ii*WIDTH_div_PACK, row_len_weight, pre_calc_n, pre_calc_m);
                    matmul_tile(output_buf_1, input_buf_1, input_scale_buf_1, weight_buf_1, weight_scale_buf_1, now_calc_n, pre_calc_m);
                    load_weight_scale_buf(weight_scale_buf_0, scale_bus, scale_base_offset, jj<<MATMUL_DEPTH_LOG2, ii*WIDTH_div_GS_PACK, row_len_scale, pre_calc_n, pre_calc_m);
                    load_input_scale_buf(input_scale_buf_0, input_scale, ii*WIDTH_div_GS, pre_calc_m);
                    load_input_buf(input_buf_0, input, ii<<MATMUL_WIDTH_LOG2, pre_calc_m);
                    save_output_buf(output, output_buf_0, jjj<<MATMUL_DEPTH_LOG2, iii==0, post_calc_n);
                }
                pingpong = 0;

            }

        }
    }
    if(pingpong == 0) {
        save_output_buf(output, output_buf_1, (loop_n-1)<<MATMUL_DEPTH_LOG2, false, now_calc_n);
    } else {
        save_output_buf(output, output_buf_0, (loop_n-1)<<MATMUL_DEPTH_LOG2, false, now_calc_n);
    }

}

void load_matmul_top_buf(
    DataType_quantize xq_buf[hidden_dim],
    DataType_scale xq_scale_buf[hidden_dim>>GS_LOG2],
    DataType_quantize* xq,
    DataType_scale* xq_scale,
    int n
) {
    #pragma HLS inline off

    const int num_groups = n >> GS_LOG2;
    load_matmul_top_buf_xq_scale:
    for(int i = 0; i < num_groups; i++) {
        #pragma HLS PIPELINE II=1
        xq_scale_buf[i] = xq_scale[i];
        for(int j = 0; j < 32; j++) {
            #pragma HLS UNROLL
            xq_buf[(i<<5)+j] = xq[(i<<5)+j];
        }
    }
}

void save_matmul_top_buf(
    DataType x_buf[hidden_dim],
    DataType* x,
    int n
) {
    #pragma HLS inline off
    const int num_groups = n >> 5;
    save_matmul_top_buf_loop:
    for(int i = 0; i < num_groups; i++) {
        #pragma HLS PIPELINE II=1
        for(int j = 0; j < 32; j++) {
            #pragma HLS UNROLL
            x[(i<<5)+j] = x_buf[(i<<5)+j];
        }
    }
}

extern "C" void matmul_top(
    DecoderBuffer& decoder_buffer,
    WeightPort& weight_port,
    MaxiOffset& maxi_offset,
    Controller* controller
) {
    #pragma HLS inline off
    DataType_quantize xq_buf[hidden_dim];
    #pragma HLS ARRAY_RESHAPE variable=xq_buf dim=1 cyclic factor=32
    #pragma HLS BIND_STORAGE variable=xq_buf type=ram_2p impl=uram
    DataType_scale xq_scale_buf[hidden_dim>>GS_LOG2];
    #pragma HLS BIND_STORAGE variable=xq_scale_buf type=ram_2p impl=uram
    DataType x_buf[vocab_size];
    #pragma HLS BIND_STORAGE variable=x_buf type=ram_2p impl=uram

    DataType_quantize_pack_maxi* weight_pack_bus1 = &weight_port.weight_bus_1;
    DataType_quantize_pack_maxi* weight_pack_bus2 = &weight_port.weight_bus_2;
    DataType_quantize_pack_maxi* weight_pack_bus3 = &weight_port.weight_bus_3;
    DataType_quantize_pack_maxi* weight_pack_bus4 = &weight_port.weight_bus_4;
    DataType_quantize_pack_maxi* weight_pack_bus5 = &weight_port.weight_bus_5;
    DataType_quantize_pack_maxi* weight_pack_bus6 = &weight_port.weight_bus_6;
    DataType_quantize_pack_maxi* weight_pack_bus7 = &weight_port.weight_bus_7;
    DataType_quantize_pack_maxi* weight_pack_bus8 = &weight_port.weight_bus_8;
    DataType_quantize_pack_maxi* weight_pack_bus9 = &weight_port.weight_bus_9;
    DataType_quantize_pack_maxi* weight_pack_bus10 = &weight_port.weight_bus_10;
    DataType_quantize_pack_maxi* weight_pack_bus11 = &weight_port.weight_bus_11;
    DataType_quantize_pack_maxi* weight_pack_bus12 = &weight_port.weight_bus_12;
    DataType_quantize_pack_maxi* weight_pack_bus13 = &weight_port.weight_bus_13;
    DataType_quantize_pack_maxi* weight_pack_bus14 = &weight_port.weight_bus_14;
    DataType_quantize_pack_maxi* weight_pack_bus15 = &weight_port.weight_bus_15;
    DataType_quantize_pack_maxi* weight_pack_bus16 = &weight_port.weight_bus_16;
    DataType_scale_pack_maxi* scale_bus = &weight_port.scale_bus;

    int weight_base_offset;
    int scale_base_offset;

    if(controller->run_matmul == RUN) {

        if(controller->index_matmul == MATMUL_Q_GENERATION) {
            load_matmul_top_buf(xq_buf, xq_scale_buf, decoder_buffer.xq, decoder_buffer.xq_scale, dim);
            weight_base_offset = maxi_offset.weight_q_pack_base_offset;
            scale_base_offset = maxi_offset.scale_q_pack_base_offset;
            matmul(x_buf, xq_buf, xq_scale_buf, *weight_pack_bus1, *weight_pack_bus2, *weight_pack_bus3, *weight_pack_bus4, *weight_pack_bus5, *weight_pack_bus6, *weight_pack_bus7, *weight_pack_bus8, *weight_pack_bus9, *weight_pack_bus10, *weight_pack_bus11, *weight_pack_bus12, *weight_pack_bus13, *weight_pack_bus14, *weight_pack_bus15, *weight_pack_bus16, weight_base_offset, *scale_bus, scale_base_offset, dim, dim);
            save_matmul_top_buf(x_buf, decoder_buffer.q, dim);
            controller->index_matmul = MATMUL_K_GENERATION;
        }
        else if(controller->index_matmul == MATMUL_K_GENERATION) {
            load_matmul_top_buf(xq_buf, xq_scale_buf, decoder_buffer.xq, decoder_buffer.xq_scale, dim);
            weight_base_offset = maxi_offset.weight_k_pack_base_offset;
            scale_base_offset = maxi_offset.scale_k_pack_base_offset;
            matmul(x_buf, xq_buf, xq_scale_buf, *weight_pack_bus1, *weight_pack_bus2, *weight_pack_bus3, *weight_pack_bus4, *weight_pack_bus5, *weight_pack_bus6, *weight_pack_bus7, *weight_pack_bus8, *weight_pack_bus9, *weight_pack_bus10, *weight_pack_bus11, *weight_pack_bus12, *weight_pack_bus13, *weight_pack_bus14, *weight_pack_bus15, *weight_pack_bus16, weight_base_offset, *scale_bus, scale_base_offset, kv_dim, dim);
            save_matmul_top_buf(x_buf, decoder_buffer.k, kv_dim);
            controller->index_matmul = MATMUL_V_GENERATION;
        }
        else if(controller->index_matmul == MATMUL_V_GENERATION) {
            load_matmul_top_buf(xq_buf, xq_scale_buf, decoder_buffer.xq, decoder_buffer.xq_scale, dim);
            weight_base_offset = maxi_offset.weight_v_pack_base_offset;
            scale_base_offset = maxi_offset.scale_v_pack_base_offset;
            matmul(x_buf, xq_buf, xq_scale_buf, *weight_pack_bus1, *weight_pack_bus2, *weight_pack_bus3, *weight_pack_bus4, *weight_pack_bus5, *weight_pack_bus6, *weight_pack_bus7, *weight_pack_bus8, *weight_pack_bus9, *weight_pack_bus10, *weight_pack_bus11, *weight_pack_bus12, *weight_pack_bus13, *weight_pack_bus14, *weight_pack_bus15, *weight_pack_bus16, weight_base_offset, *scale_bus, scale_base_offset, kv_dim, dim);
            save_matmul_top_buf(x_buf, decoder_buffer.v, kv_dim);
            controller->run_rope = RUN;
            controller->run_matmul = LAZY;
        }
        else if(controller->index_matmul == MATMUL_ATTENTION_OUTPUT) {
            load_matmul_top_buf(xq_buf, xq_scale_buf, decoder_buffer.xq, decoder_buffer.xq_scale, dim);
            weight_base_offset = maxi_offset.weight_1_pack_base_offset;
            scale_base_offset = maxi_offset.scale_1_pack_base_offset;
            matmul(x_buf, xq_buf, xq_scale_buf, *weight_pack_bus1, *weight_pack_bus2, *weight_pack_bus3, *weight_pack_bus4, *weight_pack_bus5, *weight_pack_bus6, *weight_pack_bus7, *weight_pack_bus8, *weight_pack_bus9, *weight_pack_bus10, *weight_pack_bus11, *weight_pack_bus12, *weight_pack_bus13, *weight_pack_bus14, *weight_pack_bus15, *weight_pack_bus16, weight_base_offset, *scale_bus, scale_base_offset, dim, dim);
            save_matmul_top_buf(x_buf, decoder_buffer.xb2, dim);
            controller->index_accumulate = ACCUMULATE_ATTENTION_OUTPUT;
            controller->run_accumulate = RUN;
            controller->run_matmul = LAZY;
        }
        else if(controller->index_matmul == MATMUL_GATE) {
            load_matmul_top_buf(xq_buf, xq_scale_buf, decoder_buffer.xq, decoder_buffer.xq_scale, dim);
            weight_base_offset = maxi_offset.weight_1_pack_base_offset;
            scale_base_offset = maxi_offset.scale_1_pack_base_offset;
            matmul(x_buf, xq_buf, xq_scale_buf, *weight_pack_bus1, *weight_pack_bus2, *weight_pack_bus3, *weight_pack_bus4, *weight_pack_bus5, *weight_pack_bus6, *weight_pack_bus7, *weight_pack_bus8, *weight_pack_bus9, *weight_pack_bus10, *weight_pack_bus11, *weight_pack_bus12, *weight_pack_bus13, *weight_pack_bus14, *weight_pack_bus15, *weight_pack_bus16, weight_base_offset, *scale_bus, scale_base_offset, hidden_dim, dim);
            save_matmul_top_buf(x_buf, decoder_buffer.hb, hidden_dim);
            controller->index_matmul = MATMUL_UP;
        }
        else if(controller->index_matmul == MATMUL_UP) {
            load_matmul_top_buf(xq_buf, xq_scale_buf, decoder_buffer.xq, decoder_buffer.xq_scale, dim);
            weight_base_offset = maxi_offset.weight_3_pack_base_offset;
            scale_base_offset = maxi_offset.scale_3_pack_base_offset;
            matmul(x_buf, xq_buf, xq_scale_buf, *weight_pack_bus1, *weight_pack_bus2, *weight_pack_bus3, *weight_pack_bus4, *weight_pack_bus5, *weight_pack_bus6, *weight_pack_bus7, *weight_pack_bus8, *weight_pack_bus9, *weight_pack_bus10, *weight_pack_bus11, *weight_pack_bus12, *weight_pack_bus13, *weight_pack_bus14, *weight_pack_bus15, *weight_pack_bus16, weight_base_offset, *scale_bus, scale_base_offset, hidden_dim, dim);
            save_matmul_top_buf(x_buf, decoder_buffer.hb2, hidden_dim);
            controller->run_swiglu = RUN;
            controller->run_matmul = LAZY;
        }
        else if(controller->index_matmul == MATMUL_DOWN) {
            load_matmul_top_buf(xq_buf, xq_scale_buf, decoder_buffer.hq, decoder_buffer.hq_scale, hidden_dim);
            weight_base_offset = maxi_offset.weight_2_pack_base_offset;
            scale_base_offset = maxi_offset.scale_2_pack_base_offset;
            matmul(x_buf, xq_buf, xq_scale_buf, *weight_pack_bus1, *weight_pack_bus2, *weight_pack_bus3, *weight_pack_bus4, *weight_pack_bus5, *weight_pack_bus6, *weight_pack_bus7, *weight_pack_bus8, *weight_pack_bus9, *weight_pack_bus10, *weight_pack_bus11, *weight_pack_bus12, *weight_pack_bus13, *weight_pack_bus14, *weight_pack_bus15, *weight_pack_bus16, weight_base_offset, *scale_bus, scale_base_offset, dim, hidden_dim);
            save_matmul_top_buf(x_buf, decoder_buffer.xb, dim);
            controller->index_accumulate = ACCUMULATE_FFN_OUTPUT;
            controller->run_accumulate = RUN;
            controller->run_matmul = LAZY;
        }
        else if(controller->index_matmul == MATMUL_LOGITS) {
            load_matmul_top_buf(xq_buf, xq_scale_buf, decoder_buffer.xq, decoder_buffer.xq_scale, dim);
            weight_base_offset = maxi_offset.weight_logits_pack_base_offset;
            scale_base_offset = maxi_offset.scale_logits_pack_base_offset;
            matmul(x_buf, xq_buf, xq_scale_buf, *weight_pack_bus1, *weight_pack_bus2, *weight_pack_bus3, *weight_pack_bus4, *weight_pack_bus5, *weight_pack_bus6, *weight_pack_bus7, *weight_pack_bus8, *weight_pack_bus9, *weight_pack_bus10, *weight_pack_bus11, *weight_pack_bus12, *weight_pack_bus13, *weight_pack_bus14, *weight_pack_bus15, *weight_pack_bus16, weight_base_offset, *scale_bus, scale_base_offset, vocab_size, dim);
            save_matmul_top_buf(x_buf, decoder_buffer.logits, vocab_size);
            controller->run_matmul = LAZY;
            controller->exit = RUN;
        }
    }
}
