# FT-Transformer: resilient and reliable transformer with end-to-end fault tolerant attention
Thank you for your interest! We are currently organizing the content and will update soon. Stay tuned!  

## End-to-end fault tolerant attention design
- **End-to-end fault tolerance for attention mechanism**: Error detection and correction within a fused attention kernel, reducing redundant data access and thereby mitigating memory faults; Efficient fault tolerance by avoiding unnecessary kernel launches when protecting individual operations.
- **Hybrid scheme covering all error scenarios**: Utilize tensor checksum-based ABFT to protect linear operations and selective neuron value restriction for non-linear operations. Employ a checksum reuse strategy to streamline multiple computation steps into a single verification process.
- **Architecture-aware ABFT optimized for Tensor Cores**: Leverage the thread-data mapping of matrix multiply-accumulate (MMA) instructions to perform strided computations, enabling intra-thread checksum encoding and error correction.

### A. Block-level design 
<img src="./assets/ft-attn%20arch(e-to-e)_final_acm.png" width="700">

### B. Fault tolerance workflow
<img src="./assets/ft-attn%20workload(e-to-e)_acm.png" width="680">

## Environment requirements
- CUDA: >= 12.4
- CMake: >= 3.12
- GCC: >= 5
- PyTorch: >= 2.3.0 

Tested on: A100 + CUDA 12.4 + PyTorch 2.3.0 + Python 3.9.19
