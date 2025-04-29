/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include "common.h"
#include "flash.h"
#include "flash_fwd_kernel.h"
#include "static_switch.h"

template <typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K>
__global__ void flash_fwd_kernel(Flash_fwd_params params) {
    flash::compute_attn<Kernel_traits, Is_causal, Is_even_MN, Is_even_K>(params);
}

template <typename Kernel_traits, bool Is_causal>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize; // :: 用于访问结构体中的静态成员变量，或成员函数
    // printf("smem_size = %d\n", smem_size);

    // Work-around for gcc 7. It doesn't like nested BOOL_SWITCH.
    // https://github.com/kokkos/kokkos-kernels/issues/349
    // https://github.com/HazyResearch/flash-attention/issues/21

    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr &&
                            params.seqlen_k % Kernel_traits::kBlockN == 0 &&
                            params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    // 推测：params.cu_seqlens_q == nullptr 判断结果为true，可能是因为这是一个指向gpu内存的指针，在cpu上无法访问对应空间
    // std::cout << "cu_seqlens_q: " << params.cu_seqlens_q[1] << std::endl;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        BOOL_SWITCH(is_even_K, IsEvenKConst, [&] {
            auto kernel = &flash_fwd_kernel<Kernel_traits, Is_causal, IsEvenMNConst, IsEvenKConst>; // 在C++中，函数的名称本身可以隐式转换为函数指针，但是显式加 & 能更清楚地表示这是一个函数指针
            if (smem_size >= 48 * 1024) {
                FAI_CHECK_CUDART_ERROR(
                    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            }
            int ctas_per_sm;
            FAI_CHECK_CUDART_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas_per_sm, kernel,
                                                                                 Kernel_traits::kNThreads, smem_size));
            // printf("smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);
            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
            FAI_CHECK_CUDART_ERROR(cudaPeekAtLastError());
        });
    });
}

template <typename T>
void run_mha_fwd_hdim32(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr int Headdim = 32;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_causal>(params, stream);
    });
}

template <typename T>
void run_mha_fwd_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr int Headdim = 64;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        // Using 8 warps is 18% slower for seqlen=2k, 2 warps is 5% slower
        // Using block size (64 x 256) is 27% slower for seqlen=2k
        // Using block size (256 x 64) is 85% slower for seqlen=2k, because of register spilling
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_causal>(params, stream);
    });
}

template <typename T>
void run_mha_fwd_hdim96(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr int Headdim = 96;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        // For sm86 or sm89, 64 x 64 is the fastest for causal (because it's square),
        if (params.is_sm8x) {
            if constexpr (!Is_causal) {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_causal>(params, stream);
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_causal>(params, stream);
            }
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_causal>(params, stream);
        }
    });
}

template <typename T>
void run_mha_fwd_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr int Headdim = 128;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        // For sm86 or sm89, 64 x 64 is the fastest for causal (because it's square),
        // and 128 x 32 (48 KB smem) is the fastest for non-causal since we get 2 CTAs per SM.
        if (params.is_sm8x) {
            if constexpr (!Is_causal) {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>, Is_causal>(params, stream);
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_causal>(params, stream);
            }
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_causal>(params, stream);
        }
    });
}

template <typename T>
void run_mha_fwd_hdim160(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr int Headdim = 160;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        // For A100, H100, 128 x 32 is the fastest.
        // For sm86 or sm89, 64 x 64 is the fastest for causal (because it's square),
        // and 128 x 64 with 8 warps is the fastest for non-causal.
        if (params.is_sm8x) {
            if constexpr (!Is_causal) {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_causal>(params, stream);
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_causal>(params, stream);
            }
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>, Is_causal>(params, stream);
        }
    });
}

template <typename T>
void run_mha_fwd_hdim192(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr int Headdim = 192;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_causal>(params, stream);
    });
}

template <typename T>
void run_mha_fwd_hdim224(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr int Headdim = 224;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        if (params.props->sharedMemPerBlock >= 2 * Headdim * (128 + 2 * 64)) {  // 112 KB
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_causal>(params, stream);
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_causal>(params, stream);
        }
    });
}

template <typename T>
void run_mha_fwd_hdim256(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr int Headdim = 256;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        // For A100, we want to run with 128 x 64 (128KB smem).
        // For H100 we want to run with 64 x 64 (96KB smem) since then we can get 2 CTAs per SM.
        if (params.props->sharedMemPerBlock >= 2 * Headdim * (128 + 2 * 64) &&
            params.props->sharedMemPerMultiprocessor < 4 * Headdim * (64 + 2 * 64)) {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_causal>(params, stream);
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_causal>(params, stream);
        }
    });
}
