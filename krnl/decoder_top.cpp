#include "../include/decoder_top.h"

extern "C" void decoder_top(
    int pos,
    int layer,
    DecoderBuffer& decoder_buffer,
    WeightPort& weight_port,
    MaxiOffset& maxi_offset
) {
    Controller controller;
    control_llama2_init(&controller, layer);

    decoder_loop:
    while(1) {
        if(controller.exit == RUN) break;
        nonlinear_top(decoder_buffer, weight_port, maxi_offset, &controller);
        quantize_top(decoder_buffer, &controller);
        matmul_top(decoder_buffer, weight_port, maxi_offset, &controller);
        multiattention_top(decoder_buffer, weight_port, maxi_offset, pos, &controller);
        accumulate_top(decoder_buffer, &controller);
    }
}
