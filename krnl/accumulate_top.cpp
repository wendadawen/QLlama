#include "../include/accumulate_top.h"

void accumlate(
    DataType* output,
    DataType* input
) {
    #pragma HLS inline off
    accumlate_loop:
    for(int i = 0; i < dim; i++) {
        #pragma HLS PIPELINE II=1
        output[i] += input[i];
    }
}

extern "C" void accumulate_top(
    DecoderBuffer& decoder_buffer,
    Controller* controller
) {
	#pragma HLS INLINE off
    if(controller->run_accumulate == RUN) {
        if(controller->index_accumulate == ACCUMULATE_ATTENTION_OUTPUT) {
            accumlate(decoder_buffer.x, decoder_buffer.xb2);
            controller->index_rmsnorm = RMSNORM_FFN;
            controller->run_rmsnorm = RUN;
            controller->run_accumulate = LAZY;
        }
        else if(controller->index_accumulate == ACCUMULATE_FFN_OUTPUT) {
            accumlate(decoder_buffer.x, decoder_buffer.xb);
            controller->exit = RUN;
            controller->run_accumulate = LAZY;
        }
    }
}
