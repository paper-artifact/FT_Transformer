#ifndef __FLASH_ATTENTION_INFERENCE_CUDA_TIMER_H__
#define __FLASH_ATTENTION_INFERENCE_CUDA_TIMER_H__

#include "common.h"

class CudaTimer {
public:
    CudaTimer() {
        FAI_CHECK_CUDART_ERROR(cudaEventCreate(&m_start));
        FAI_CHECK(m_start);
        FAI_CHECK_CUDART_ERROR(cudaEventCreate(&m_end));
        FAI_CHECK(m_end);
    }

    ~CudaTimer() {
        if (m_start) {
            FAI_CHECK_CUDART_ERROR(cudaEventDestroy(m_start));
            m_start = nullptr;
        }

        if (m_end) {
            FAI_CHECK_CUDART_ERROR(cudaEventDestroy(m_end));
            m_end = nullptr;
        }
    }

    void start() {
        FAI_CHECK_CUDART_ERROR(cudaEventRecord(m_start));
    }

    float end() {
        FAI_CHECK_CUDART_ERROR(cudaEventRecord(m_end));
        FAI_CHECK_CUDART_ERROR(cudaEventSynchronize(m_end));
        FAI_CHECK_CUDART_ERROR(cudaEventElapsedTime(&m_elapsed_time, m_start, m_end));

        return m_elapsed_time;
    }

private:
    cudaEvent_t m_start = nullptr;
    cudaEvent_t m_end = nullptr;
    float m_elapsed_time = 0.0;

    FAI_DISALLOW_COPY_AND_ASSIGN(CudaTimer);
};

#endif  // __FLASH_ATTENTION_INFERENCE_CUDA_TIMER_H__
