#!/bin/bash

set -euo pipefail

WORK_PATH=$(cd $(dirname $0) && pwd) && cd $WORK_PATH

profile_Base_Huge(){
    echo "Profiling the basic end-to-end ft attention, head=32, dimension=128"

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=32 -sq=512 -sk=512 -hq=32 -hk=32 -d=128 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_baseline_paper/etoe_base_huge_32_512_512_32_32_128_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=16 -sq=1024 -sk=1024 -hq=32 -hk=32 -d=128 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_baseline_paper/etoe_base_huge_16_1024_1024_32_32_128_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=8 -sq=2048 -sk=2048 -hq=32 -hk=32 -d=128 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_baseline_paper/etoe_base_huge_8_2048_2048_32_32_128_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=4 -sq=4096 -sk=4096 -hq=32 -hk=32 -d=128 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_baseline_paper/etoe_base_huge_4_4096_4096_32_32_128_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=2 -sq=8192 -sk=8192 -hq=32 -hk=32 -d=128 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_baseline_paper/etoe_base_huge_2_8192_8192_32_32_128_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=1 -sq=16384 -sk=16384 -hq=32 -hk=32 -d=128 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_baseline_paper/etoe_base_huge_1_16384_16384_32_32_128_A100_40G.log 2>&1 &
}

profile_Base_Small(){
    echo "Profiling the basic end-to-end ft attention, head=16, dimension=64"

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=32 -sq=512 -sk=512 -hq=16 -hk=16 -d=64 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_baseline_paper/etoe_base_small_32_512_512_16_16_64_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=16 -sq=1024 -sk=1024 -hq=16 -hk=16 -d=64 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_baseline_paper/etoe_base_small_16_1024_1024_16_16_64_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=8 -sq=2048 -sk=2048 -hq=16 -hk=16 -d=64 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_baseline_paper/etoe_base_small_8_2048_2048_16_16_64_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=4 -sq=4096 -sk=4096 -hq=16 -hk=16 -d=64 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_baseline_paper/etoe_base_small_4_4096_4096_16_16_64_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=2 -sq=8192 -sk=8192 -hq=16 -hk=16 -d=64 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_baseline_paper/etoe_base_small_2_8192_8192_16_16_64_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=1 -sq=16384 -sk=16384 -hq=16 -hk=16 -d=64 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_baseline_paper/etoe_base_small_1_16384_16384_16_16_64_A100_40G.log 2>&1 &
}

Profile_ABFT_Huge(){
    echo "Profiling the end-to-end ft attention with strided ABFT, head=32, dimension=128"

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=32 -sq=512 -sk=512 -hq=32 -hk=32 -d=128 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_ABFT_all_paper/etoe_ABFT_all_huge_32_512_512_32_32_128_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=16 -sq=1024 -sk=1024 -hq=32 -hk=32 -d=128 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_ABFT_all_paper/etoe_ABFT_all_huge_16_1024_1024_32_32_128_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=8 -sq=2048 -sk=2048 -hq=32 -hk=32 -d=128 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_ABFT_all_paper/etoe_ABFT_all_huge_8_2048_2048_32_32_128_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=4 -sq=4096 -sk=4096 -hq=32 -hk=32 -d=128 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_ABFT_all_paper/etoe_ABFT_all_huge_4_4096_4096_32_32_128_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=2 -sq=8192 -sk=8192 -hq=32 -hk=32 -d=128 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_ABFT_all_paper/etoe_ABFT_all_huge_2_8192_8192_32_32_128_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=1 -sq=16384 -sk=16384 -hq=32 -hk=32 -d=128 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_ABFT_all_paper/etoe_ABFT_all_huge_1_16384_16384_32_32_128_A100_40G.log 2>&1 &
}

Profile_ABFT_Small(){
    echo "Profiling the end-to-end ft attention with strided ABFT, head=16, dimension=64"

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=32 -sq=512 -sk=512 -hq=16 -hk=16 -d=64 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_ABFT_all_paper/etoe_ABFT_all_small_32_512_512_32_32_128_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=16 -sq=1024 -sk=1024 -hq=16 -hk=16 -d=64 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_ABFT_all_paper/etoe_ABFT_all_small_16_1024_1024_32_32_128_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=8 -sq=2048 -sk=2048 -hq=16 -hk=16 -d=64 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_ABFT_all_paper/etoe_ABFT_all_small_8_2048_2048_32_32_128_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=4 -sq=4096 -sk=4096 -hq=16 -hk=16 -d=64 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_ABFT_all_paper/etoe_ABFT_all_small_4_4096_4096_32_32_128_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=2 -sq=8192 -sk=8192 -hq=16 -hk=16 -d=64 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_ABFT_all_paper/etoe_ABFT_all_small_2_8192_8192_32_32_128_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=1 -sq=16384 -sk=16384 -hq=16 -hk=16 -d=64 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_ABFT_all_paper/etoe_ABFT_all_small_1_16384_16384_32_32_128_A100_40G.log 2>&1 &
}

Profile_SNVR_Huge(){
    echo "Profiling the end-to-end ft attention with selective neuron value restriction, head=32, dimension=128"

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=32 -sq=512 -sk=512 -hq=32 -hk=32 -d=128 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_selective_paper/etoe_selective_huge_32_512_512_32_32_128_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=16 -sq=1024 -sk=1024 -hq=32 -hk=32 -d=128 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_selective_paper/etoe_selective_huge_16_1024_1024_32_32_128_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=8 -sq=2048 -sk=2048 -hq=32 -hk=32 -d=128 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_selective_paper/etoe_selective_huge_8_2048_2048_32_32_128_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=4 -sq=4096 -sk=4096 -hq=32 -hk=32 -d=128 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_selective_paper/etoe_selective_huge_4_4096_4096_32_32_128_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=2 -sq=8192 -sk=8192 -hq=32 -hk=32 -d=128 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_selective_paper/etoe_selective_huge_2_8192_8192_32_32_128_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=1 -sq=16384 -sk=16384 -hq=32 -hk=32 -d=128 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_selective_paper/etoe_selective_huge_1_16384_16384_32_32_128_A100_40G.log 2>&1 &
}

Profile_SNVR_Small(){
    echo "Profiling the end-to-end ft attention with selective neuron value restriction, head=16, dimension=64"

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=32 -sq=512 -sk=512 -hq=16 -hk=16 -d=64 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_selective_paper/etoe_selective_small_32_512_512_32_32_128_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=16 -sq=1024 -sk=1024 -hq=16 -hk=16 -d=64 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_selective_paper/etoe_selective_small_16_1024_1024_32_32_128_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=8 -sq=2048 -sk=2048 -hq=16 -hk=16 -d=64 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_selective_paper/etoe_selective_small_8_2048_2048_32_32_128_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=4 -sq=4096 -sk=4096 -hq=16 -hk=16 -d=64 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_selective_paper/etoe_selective_small_4_4096_4096_32_32_128_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=2 -sq=8192 -sk=8192 -hq=16 -hk=16 -d=64 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_selective_paper/etoe_selective_small_2_8192_8192_32_32_128_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=1 -sq=16384 -sk=16384 -hq=16 -hk=16 -d=64 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_selective_paper/etoe_selective_small_1_16384_16384_32_32_128_A100_40G.log 2>&1 &
}

Profile_Optimized_Huge(){
    echo "Profiling the optimized end-to-end ft attention, head=32, dimension=128"

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=32 -sq=512 -sk=512 -hq=32 -hk=32 -d=128 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_FT_all_optimized_paper/etoe_FT_all_optimized_huge_32_512_512_32_32_128_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=16 -sq=1024 -sk=1024 -hq=32 -hk=32 -d=128 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_FT_all_optimized_paper/etoe_FT_all_optimized_huge_16_1024_1024_32_32_128_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=8 -sq=2048 -sk=2048 -hq=32 -hk=32 -d=128 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_FT_all_optimized_paper/etoe_FT_all_optimized_huge_8_2048_2048_32_32_128_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=4 -sq=4096 -sk=4096 -hq=32 -hk=32 -d=128 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_FT_all_optimized_paper/etoe_FT_all_optimized_huge_4_4096_4096_32_32_128_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=2 -sq=8192 -sk=8192 -hq=32 -hk=32 -d=128 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_FT_all_optimized_paper/etoe_FT_all_optimized_huge_2_8192_8192_32_32_128_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=1 -sq=16384 -sk=16384 -hq=32 -hk=32 -d=128 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_FT_all_optimized_paper/etoe_FT_all_optimized_huge_1_16384_16384_32_32_128_A100_40G.log 2>&1 &
}

Profile_Optimized_Small(){
    echo "Profiling the optimized end-to-end ft attention, head=16, dimension=64"

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=32 -sq=512 -sk=512 -hq=16 -hk=16 -d=64 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_FT_all_optimized_paper/etoe_FT_all_optimized_small_32_512_512_16_16_64_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=16 -sq=1024 -sk=1024 -hq=16 -hk=16 -d=64 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_FT_all_optimized_paper/etoe_FT_all_optimized_small_16_1024_1024_16_16_64_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=8 -sq=2048 -sk=2048 -hq=16 -hk=16 -d=64 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_FT_all_optimized_paper/etoe_FT_all_optimized_small_8_2048_2048_16_16_64_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=4 -sq=4096 -sk=4096 -hq=16 -hk=16 -d=64 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_FT_all_optimized_paper/etoe_FT_all_optimized_small_4_4096_4096_16_16_64_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=2 -sq=8192 -sk=8192 -hq=16 -hk=16 -d=64 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_FT_all_optimized_paper/etoe_FT_all_optimized_small_2_8192_8192_16_16_64_A100_40G.log 2>&1 &
    PID=$!               
    wait $PID

    nohup $WORK_PATH/output/bin/flash_attention_inference -b=1 -sq=16384 -sk=16384 -hq=16 -hk=16 -d=64 -is_causal=false -num_splits=0 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > exp_final/etoe_FT_all_optimized_paper/etoe_FT_all_optimized_small_1_16384_16384_16_16_64_A100_40G.log 2>&1 &
}

param_s=""
param_v=""


while getopts "s:v:" opt
do
    case "$opt" in
        s) param_s="$OPTARG" ;;
        v) param_v="$OPTARG" ;;
        *)
            echo "Usage: $0 -s <Huge, Small> -v <Basic, ABFT, SNVR, Optimized>"
            exit 1
            ;;
    esac
done

if [ -z "$param_s" ] || [ -z "$param_v" ]; then
    echo "Both -s and -v options are required."
    echo "Usage: $0 -s <Huge, Small> -v <Basic, ABFT, SNVR, Optimized>"
    exit 1
fi

case "$param_s" in
    Huge)
        case "$param_v" in
            Basic)
                profile_Base_Huge
                ;;
            ABFT)
                Profile_ABFT_Huge
                ;;
            SNVR)
                Profile_SNVR_Huge
                ;;
            Optimized)
                Profile_Optimized_Huge
                ;;
            *)
                echo "Invalid value for -v: $param_v"
                exit 1
                ;;
        esac
        ;;
    Small)
        case "$param_v" in
            Basic)
                profile_Base_Small
                ;;
            ABFT)
                Profile_ABFT_Small
                ;;
            SNVR)
                Profile_SNVR_Small
                ;;
            optimized)
                Profile_Optimized_Small
                ;;
            *)
                echo "Invalid value for -v: $param_v"
                exit 1
                ;;
        esac
        ;;
    *)
        echo "Invalid value for -s: $param_s"
        exit 1
        ;;
esac




