#include "config.h"

extern "C" void multiattention_top(
    DecoderBuffer& decoder_buffer,
    WeightPort& weight_port,
    MaxiOffset& maxi_offset,
    int pos,
    Controller* controller
);
