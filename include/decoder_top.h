#pragma once
#include "config.h"
#include "nonlinear_top.h"
#include "quantize_top.h"
#include "matmul_top.h"
#include "multiattention_top.h"
#include "accumulate_top.h"

extern "C" void decoder_top(
    int pos,
    int layer,
    DecoderBuffer& decoder_buffer,
    WeightPort& weight_port,
    MaxiOffset& maxi_offset
);
