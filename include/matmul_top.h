#include "config.h"

extern "C" void matmul_top(
    DecoderBuffer& decoder_buffer,
    WeightPort& weight_port,
    MaxiOffset& maxi_offset,
    Controller* controller
);
