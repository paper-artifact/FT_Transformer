#include "gflags/gflags.h"
#include "omp.h"
#include "tester.h"

#define FLASH_ATTENTION_FUNC(name)                                                                      \
    void name(Tensor<cutlass::half_t> *Q, Tensor<cutlass::half_t> *K, Tensor<cutlass::half_t> *V,       \
              Tensor<cutlass::half_t> *O, int *cu_seq_q, int *cu_seq_k, bool is_causal, int num_splits, \
              cudaDeviceProp *dev_prop)


FLASH_ATTENTION_FUNC(flash_attn_v2); 

DEFINE_uint32(b, 2, "batch size"); 
DEFINE_uint32(sq, 256, "q seq len");
DEFINE_uint32(sk, 256, "k seq len");
DEFINE_uint32(hq, 32, "q head num");
DEFINE_uint32(hk, 32, "k head num");
DEFINE_uint32(d, 128, "head dim");
DEFINE_bool(is_causal, true, "causal mask");
DEFINE_int32(num_splits, 0, "num splits of seq q len for flash attn");
DEFINE_uint32(warmup_iterations, 1, "warmup iteration numbers and average the result");
DEFINE_uint32(profiling_iterations, 10, "profiling iteration numbers and average the result");
DEFINE_uint32(sleep_duration, 100, "sleep_milliseconds between profiling");
DEFINE_bool(enable_check, false, "check the GPU result against the CPU result");
DEFINE_uint32(cpu_procs, omp_get_num_procs(), "processor num used of CPU");
DEFINE_uint32(gpu_rank, 0, "the used GPU rank");

int main(int argc, char *argv[]) {
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    omp_set_num_threads(FLAGS_cpu_procs);
    FAI_CHECK_CUDART_ERROR(cudaSetDevice(FLAGS_gpu_rank));

    cudaDeviceProp dev_prop;
    FAI_CHECK_CUDART_ERROR(cudaGetDeviceProperties(&dev_prop, FLAGS_gpu_rank));
    FLOG("Flash Attention start with %u CPU processes on the %u-th GPU: %s", FLAGS_cpu_procs, FLAGS_gpu_rank,
         dev_prop.name);

    int driver_version = 0;
    int runtime_version = 0;
    FAI_CHECK_CUDART_ERROR(cudaDriverGetVersion(&driver_version));
    FAI_CHECK_CUDART_ERROR(cudaRuntimeGetVersion(&runtime_version));
    FLOG("CUDA driver version / runtime version: %d.%d / %d.%d", driver_version / 1000, (driver_version % 100) / 10,
         runtime_version / 1000, (runtime_version % 100) / 10);

    FLOG(
        "MHA: Softmax (Q (%u x %u x %u x %u) * K^T (%u x %u x %u x %u)) * V (%u x %u x %u x %u) = O (%u x %u x %u x "
        "%u)",
        FLAGS_b, FLAGS_sq, FLAGS_hq, FLAGS_d, FLAGS_b, FLAGS_sk, FLAGS_hk, FLAGS_d, FLAGS_b, FLAGS_sk, FLAGS_hk,
        FLAGS_d, FLAGS_b, FLAGS_sq, FLAGS_hq, FLAGS_d);
    FLOG(
        "Profiling: is causal %d, num splits: %d, warmup iterations: %u, profiling iterations: %u, sleep "
        "duration: %u ms, enable check: %d",
        FLAGS_is_causal, FLAGS_num_splits, FLAGS_warmup_iterations, FLAGS_profiling_iterations, FLAGS_sleep_duration,
        FLAGS_enable_check);

    Tester tester(FLAGS_b, FLAGS_sq, FLAGS_sk, FLAGS_hq, FLAGS_hk, FLAGS_d, FLAGS_is_causal, FLAGS_num_splits,
                  &dev_prop, FLAGS_warmup_iterations, FLAGS_profiling_iterations, FLAGS_sleep_duration,
                  FLAGS_enable_check);
    
    tester.evaluate(flash_attn_v2, "Flash-Attention-V2");

    GFLAGS_NAMESPACE::ShutDownCommandLineFlags();

    FLOG("Done");

    return 0;
}
