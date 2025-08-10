# QLlama

Official implementation of **QLlama: An FPGA-Based Microscaling Quantization Accelerator for Energy-Efficient Llama2 Inference**.

[Paper](https://doi.org/10.1109/LES.2025.3600563)

## Overview

QLlama is an FPGA-based accelerator for energy-efficient Llama2 inference. It combines a microscaling data format with hardware-software co-design. Each subtensor block shares an E8M0 scaling factor, allowing dequantization to be performed using only shift operations.

Mixed-precision configurations are applied across different Llama2 layers to balance accuracy and hardware efficiency. The accelerator includes dedicated units for dynamic quantization, vector-matrix multiplication, scaled dot products, density computation, and basic operators. The paper reports energy-efficiency improvements of 2.13x to 10.66x with negligible accuracy loss.

## Citation

```bibtex
@article{wen2025qllama,
  title={QLlama: An FPGA-Based Microscaling Quantization Accelerator for Energy-Efficient Llama2 Inference},
  author={Wen, Hongbing and Wang, Zihao and Dong, Jiale and Lou, Wenqi and Gong, Lei and Wang, Chao and Zhou, Xuehai},
  journal={IEEE Embedded Systems Letters},
  volume={17},
  number={5},
  pages={337--340},
  year={2025},
  doi={10.1109/LES.2025.3600563}
}
```
