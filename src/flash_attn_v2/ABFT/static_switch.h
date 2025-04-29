// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```
#define BOOL_SWITCH(COND, CONST_NAME, ...)            \
    [&] {                                             \
        if (COND) {                                   \
            constexpr static bool CONST_NAME = true;  \
            return __VA_ARGS__();                     \
        } else {                                      \
            constexpr static bool CONST_NAME = false; \
            return __VA_ARGS__();                     \
        }                                             \
    }()

// 在编译时运行，判断应该使用的特化模板
#define FWD_HEADDIM_SWITCH(HEADDIM, ...)         \
    [&] {                                        \
        if (HEADDIM <= 32) {                     \
            constexpr static int kHeadDim = 32;  \
            return __VA_ARGS__();                \
        } else if (HEADDIM <= 64) {              \
            constexpr static int kHeadDim = 64;  \
            return __VA_ARGS__();                \
        } else if (HEADDIM <= 96) {              \
            constexpr static int kHeadDim = 96;  \
            return __VA_ARGS__();                \
        } else if (HEADDIM <= 128) {             \
            constexpr static int kHeadDim = 128; \
            return __VA_ARGS__();                \
        } else if (HEADDIM <= 160) {             \
            constexpr static int kHeadDim = 160; \
            return __VA_ARGS__();                \
        } else if (HEADDIM <= 192) {             \
            constexpr static int kHeadDim = 192; \
            return __VA_ARGS__();                \
        } else if (HEADDIM <= 224) {             \
            constexpr static int kHeadDim = 224; \
            return __VA_ARGS__();                \
        } else if (HEADDIM <= 256) {             \
            constexpr static int kHeadDim = 256; \
            return __VA_ARGS__();                \
        }                                        \
    }() // 此处()表示立即调用lambda表达式
    // __VA_ARGS__ 是一个预处理器宏参数，表示可变参数的内容。它允许在宏定义中接受可变数量的参数
    // 在宏定义中，... 用于声明宏接受任意数量的参数，__VA_ARGS__ 用来替代这些参数
    // 当宏中的可变参数 __VA_ARGS__ 表示一组代码或表达式时，后面加上 () 表示将这些代码作为函数调用
