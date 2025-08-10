#include "../include/quantize_top.h"

void quantize_max(
    hls::stream<DataType>& x_stream,
    hls::stream<DataType_quantize>& xq_stream,
    hls::stream<int>& exp_stream,
    hls::stream<int>& max_exp_stream
) {
    #pragma HLS inline off
    int max_exp = -255;

    quantize_max_loop:
    for(int i = 0; i < GS; i++) {
        #pragma HLS PIPELINE II=1
        DataType x_val = x_stream.read();
        float a,b;
        int exp_i;
        a = (float)x_val;
        b = frexp(a, &exp_i);
        xq_stream.write((DataType_quantize)b);
        exp_stream.write(exp_i);
        if(exp_i > max_exp) max_exp = exp_i;
    }
    if(max_exp < -127) max_exp = -127;
    max_exp_stream.write(max_exp);
}

void quantize_scale(
    hls::stream<DataType_quantize>& xq_stream,
    hls::stream<int>& exp_stream,
    hls::stream<int>& max_exp_stream,
    hls::stream<DataType_quantize>& xq_out_stream,
    hls::stream<int>& max_exp_out_stream
) {
    #pragma HLS inline off
    int max_exp = max_exp_stream.read();

    quantize_scale_loop:
    for(int i = 0; i < GS; i++) {
        #pragma HLS PIPELINE II=1
        DataType_quantize xq_val = xq_stream.read();
        int exp_i = exp_stream.read();
        int dif = max_exp - exp_i;
        xq_out_stream.write(xq_val >> dif);
    }
    max_exp_out_stream.write(max_exp + 127);
}

void load_x_to_stream(
    DataType* x,
    int n,
    hls::stream<DataType>& x_stream,
    hls::stream<int>& group_stream
) {
    #pragma HLS inline off
    load_x_to_stream_loop:
    for(int i = 0; i < n; i++) {
        #pragma HLS PIPELINE II=1
        x_stream.write(x[i]);
        if((i & (GS - 1)) == 0) {
            group_stream.write(i >> GS_LOG2);
        }
    }
}

void save_x_from_stream(
    hls::stream<DataType_quantize>& xq_stream,
    hls::stream<int>& max_exp_stream,
    hls::stream<int>& group_stream,
    int n,
    DataType_quantize* xq,
    DataType_scale* xq_scale
) {
    #pragma HLS inline off
    for(int i = 0; i < n; i++) {
        #pragma HLS PIPELINE II=1
        xq[i] = xq_stream.read();
        if((i & (GS - 1)) == 0) {
            int group = group_stream.read();
            xq_scale[group] = max_exp_stream.read();
        }
    }
}

void quantize(
    DataType_quantize* xq,
    DataType_scale* xq_scale,
    DataType* x,
    int n
) {
    #pragma HLS inline off
    #pragma HLS DATAFLOW

    hls_thread_local hls::stream<DataType, 2*GS> load_to_t1_x;
    hls_thread_local hls::stream<int, 2*1> load_to_save_group;
    hls_thread_local hls::stream<DataType_quantize, 2*GS> t1_to_t2_xq;
    hls_thread_local hls::stream<int, 2*GS> t1_to_t2_exp;
    hls_thread_local hls::stream<int, 2*1> t1_to_t2_max_exp;
    hls_thread_local hls::stream<DataType_quantize, 2*GS> t2_to_save_xq;
    hls_thread_local hls::stream<int, 2*1> t2_to_save_max_exp;

    load_x_to_stream(x, n, load_to_t1_x, load_to_save_group);
    hls_thread_local hls::task t1(quantize_max, load_to_t1_x, t1_to_t2_xq, t1_to_t2_exp, t1_to_t2_max_exp);
    hls_thread_local hls::task t2(quantize_scale, t1_to_t2_xq, t1_to_t2_exp, t1_to_t2_max_exp, t2_to_save_xq, t2_to_save_max_exp);
    save_x_from_stream(t2_to_save_xq, t2_to_save_max_exp, load_to_save_group, n, xq, xq_scale);
}

extern "C" void quantize_top(
    DecoderBuffer& decoder_buffer,
    Controller* controller
) {
    #pragma HLS inline off

    if(controller->run_quantize == RUN) {

        if(controller->index_quantize == QUANTIZE_RMSNORM_ATT) {
            quantize(decoder_buffer.xq, decoder_buffer.xq_scale, decoder_buffer.xb, dim);
            controller->index_matmul = MATMUL_Q_GENERATION;
        }
        else if(controller->index_quantize == QUANTIZE_MULTIATTENTION) {
            quantize(decoder_buffer.xq, decoder_buffer.xq_scale, decoder_buffer.xb, dim);
            controller->index_matmul = MATMUL_ATTENTION_OUTPUT;
        }
        else if(controller->index_quantize == QUANTIZE_RMSNORM_FFN) {
            quantize(decoder_buffer.xq, decoder_buffer.xq_scale, decoder_buffer.xb, dim);
            controller->index_matmul = MATMUL_GATE;
        }
        else if(controller->index_quantize == QUANTIZE_SWIGLU) {
            quantize(decoder_buffer.hq, decoder_buffer.hq_scale, decoder_buffer.hb3, hidden_dim);
            controller->index_matmul = MATMUL_DOWN;
        }
        else if(controller->index_quantize == QUANTIZE_RMSNORM_FINAL) {
            quantize(decoder_buffer.xq, decoder_buffer.xq_scale, decoder_buffer.xb, dim);
            controller->index_matmul = MATMUL_LOGITS;
        }

        controller->run_quantize = LAZY;
        controller->run_matmul = RUN;
    }
}
