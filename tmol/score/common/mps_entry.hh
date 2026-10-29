#pragma once

// Buffer bindings for a Metal entry point.
//
// A context's views arrive as positional untyped buffers (see mps_launch.hh),
// so every entry spells out the same `device void* bK [[buffer(K)]]` run and
// gathers it into an array. TMOL_MPS_BIND_<N> is that run for N views and
// TMOL_MPS_PTRS_<N> the matching initializer; <N> must equal the context's
// n_views, which the entry static_asserts.

#define TMOL_MPS_BIND_1 device void* b1 [[buffer(1)]]
#define TMOL_MPS_BIND_2 TMOL_MPS_BIND_1, device void* b2 [[buffer(2)]]
#define TMOL_MPS_BIND_3 TMOL_MPS_BIND_2, device void* b3 [[buffer(3)]]
#define TMOL_MPS_BIND_4 TMOL_MPS_BIND_3, device void* b4 [[buffer(4)]]
#define TMOL_MPS_BIND_5 TMOL_MPS_BIND_4, device void* b5 [[buffer(5)]]
#define TMOL_MPS_BIND_6 TMOL_MPS_BIND_5, device void* b6 [[buffer(6)]]
#define TMOL_MPS_BIND_7 TMOL_MPS_BIND_6, device void* b7 [[buffer(7)]]
#define TMOL_MPS_BIND_8 TMOL_MPS_BIND_7, device void* b8 [[buffer(8)]]
#define TMOL_MPS_BIND_9 TMOL_MPS_BIND_8, device void* b9 [[buffer(9)]]
#define TMOL_MPS_BIND_10 TMOL_MPS_BIND_9, device void* b10 [[buffer(10)]]
#define TMOL_MPS_BIND_11 TMOL_MPS_BIND_10, device void* b11 [[buffer(11)]]
#define TMOL_MPS_BIND_12 TMOL_MPS_BIND_11, device void* b12 [[buffer(12)]]
#define TMOL_MPS_BIND_13 TMOL_MPS_BIND_12, device void* b13 [[buffer(13)]]
#define TMOL_MPS_BIND_14 TMOL_MPS_BIND_13, device void* b14 [[buffer(14)]]
#define TMOL_MPS_BIND_15 TMOL_MPS_BIND_14, device void* b15 [[buffer(15)]]
#define TMOL_MPS_BIND_16 TMOL_MPS_BIND_15, device void* b16 [[buffer(16)]]
#define TMOL_MPS_BIND_17 TMOL_MPS_BIND_16, device void* b17 [[buffer(17)]]
#define TMOL_MPS_BIND_18 TMOL_MPS_BIND_17, device void* b18 [[buffer(18)]]
#define TMOL_MPS_BIND_19 TMOL_MPS_BIND_18, device void* b19 [[buffer(19)]]
#define TMOL_MPS_BIND_20 TMOL_MPS_BIND_19, device void* b20 [[buffer(20)]]
#define TMOL_MPS_BIND_21 TMOL_MPS_BIND_20, device void* b21 [[buffer(21)]]
#define TMOL_MPS_BIND_22 TMOL_MPS_BIND_21, device void* b22 [[buffer(22)]]
#define TMOL_MPS_BIND_23 TMOL_MPS_BIND_22, device void* b23 [[buffer(23)]]
#define TMOL_MPS_BIND_24 TMOL_MPS_BIND_23, device void* b24 [[buffer(24)]]
#define TMOL_MPS_BIND_25 TMOL_MPS_BIND_24, device void* b25 [[buffer(25)]]
#define TMOL_MPS_BIND_26 TMOL_MPS_BIND_25, device void* b26 [[buffer(26)]]
#define TMOL_MPS_BIND_27 TMOL_MPS_BIND_26, device void* b27 [[buffer(27)]]
#define TMOL_MPS_BIND_28 TMOL_MPS_BIND_27, device void* b28 [[buffer(28)]]
#define TMOL_MPS_BIND_29 TMOL_MPS_BIND_28, device void* b29 [[buffer(29)]]
#define TMOL_MPS_BIND_30 TMOL_MPS_BIND_29, device void* b30 [[buffer(30)]]

#define TMOL_MPS_PTRS_1 b1
#define TMOL_MPS_PTRS_2 TMOL_MPS_PTRS_1, b2
#define TMOL_MPS_PTRS_3 TMOL_MPS_PTRS_2, b3
#define TMOL_MPS_PTRS_4 TMOL_MPS_PTRS_3, b4
#define TMOL_MPS_PTRS_5 TMOL_MPS_PTRS_4, b5
#define TMOL_MPS_PTRS_6 TMOL_MPS_PTRS_5, b6
#define TMOL_MPS_PTRS_7 TMOL_MPS_PTRS_6, b7
#define TMOL_MPS_PTRS_8 TMOL_MPS_PTRS_7, b8
#define TMOL_MPS_PTRS_9 TMOL_MPS_PTRS_8, b9
#define TMOL_MPS_PTRS_10 TMOL_MPS_PTRS_9, b10
#define TMOL_MPS_PTRS_11 TMOL_MPS_PTRS_10, b11
#define TMOL_MPS_PTRS_12 TMOL_MPS_PTRS_11, b12
#define TMOL_MPS_PTRS_13 TMOL_MPS_PTRS_12, b13
#define TMOL_MPS_PTRS_14 TMOL_MPS_PTRS_13, b14
#define TMOL_MPS_PTRS_15 TMOL_MPS_PTRS_14, b15
#define TMOL_MPS_PTRS_16 TMOL_MPS_PTRS_15, b16
#define TMOL_MPS_PTRS_17 TMOL_MPS_PTRS_16, b17
#define TMOL_MPS_PTRS_18 TMOL_MPS_PTRS_17, b18
#define TMOL_MPS_PTRS_19 TMOL_MPS_PTRS_18, b19
#define TMOL_MPS_PTRS_20 TMOL_MPS_PTRS_19, b20
#define TMOL_MPS_PTRS_21 TMOL_MPS_PTRS_20, b21
#define TMOL_MPS_PTRS_22 TMOL_MPS_PTRS_21, b22
#define TMOL_MPS_PTRS_23 TMOL_MPS_PTRS_22, b23
#define TMOL_MPS_PTRS_24 TMOL_MPS_PTRS_23, b24
#define TMOL_MPS_PTRS_25 TMOL_MPS_PTRS_24, b25
#define TMOL_MPS_PTRS_26 TMOL_MPS_PTRS_25, b26
#define TMOL_MPS_PTRS_27 TMOL_MPS_PTRS_26, b27
#define TMOL_MPS_PTRS_28 TMOL_MPS_PTRS_27, b28
#define TMOL_MPS_PTRS_29 TMOL_MPS_PTRS_28, b29
#define TMOL_MPS_PTRS_30 TMOL_MPS_PTRS_29, b30
