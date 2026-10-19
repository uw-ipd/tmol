#pragma once

#include "qualifiers.hh"

// int64_t is used below for sizes and strides. Metal's standard library
// supplies the fixed-width typedefs itself and cannot parse <cstdint>, which
// drags in host headers; every other backend needs the include, and needs it
// here rather than transitively, so this header can be included first.
#if !TMOL_PORTABLE_METAL
#include <cstdint>
#endif

namespace tmol {
namespace portable {

// A tensor view usable from a Metal kernel.
//
// tmol::TView is three pointers -- data, sizes, strides -- but it lives in
// TensorAccessor.h, which reaches torch and so cannot be included by a `.metal`
// translation unit. MetalView has the same shape and the same indexing
// semantics, with the address space Metal requires on every pointer.
//
// Kernel bodies are templated on their context type, so they index a MetalView
// exactly as they index a TView and need no change.
//
// TMOL_SHARED_PTR is the companion for pointers into workgroup-shared memory.
// An unqualified `T*` is a thread-space pointer in MSL, so the per-residue
// scratch structs -- which point into the threadgroup tile buffers -- have to
// name the address space explicitly. It expands to a plain `T*` everywhere
// else, where shared memory is just memory. TMOL_SHARED_REF does the same for
// a reference parameter -- in MSL the address space is part of a reference's
// type, so a function taking the workgroup's tile buffers by plain `T&` would
// be asking for a thread-space object and would not bind.
// TMOL_DEVICE_REF is the same thing for a reference to an element read
// straight out of a view -- a parameter table entry, say. Copying it to thread
// space first would work but costs a register-resident copy of the whole
// struct; naming the device address space lets the callee load the fields it
// uses in place.
// Variadic so that a template-id with commas in it -- which these are often
// applied to -- survives macro argument splitting.
#if TMOL_PORTABLE_METAL
#define TMOL_DEVICE_PTR(...) device __VA_ARGS__*
#define TMOL_DEVICE_REF(...) device __VA_ARGS__&
#define TMOL_SHARED_PTR(...) threadgroup __VA_ARGS__*
#define TMOL_THREAD_PTR(...) thread __VA_ARGS__*
#define TMOL_SHARED_REF(...) threadgroup __VA_ARGS__&
#else
#define TMOL_DEVICE_PTR(...) __VA_ARGS__*
#define TMOL_DEVICE_REF(...) __VA_ARGS__&
#define TMOL_SHARED_PTR(...) __VA_ARGS__*
#define TMOL_THREAD_PTR(...) __VA_ARGS__*
#define TMOL_SHARED_REF(...) __VA_ARGS__&
#endif

template <typename T, int N>
struct MetalView {
  TMOL_DEVICE_PTR(T) data_;
  TMOL_DEVICE_PTR(const int64_t) sizes_;
  TMOL_DEVICE_PTR(const int64_t) strides_;

  TMOL_HD int64_t size(int i) const { return sizes_[i]; }
  TMOL_HD int64_t stride(int i) const { return strides_[i]; }

  // Indexing a rank-N view yields a rank-(N-1) view; rank 1 yields the element.
  TMOL_HD MetalView<T, N - 1> operator[](int64_t i) const {
    return MetalView<T, N - 1>{
        data_ + strides_[0] * i, sizes_ + 1, strides_ + 1};
  }
};

template <typename T>
struct MetalView<T, 1> {
  TMOL_DEVICE_PTR(T) data_;
  TMOL_DEVICE_PTR(const int64_t) sizes_;
  TMOL_DEVICE_PTR(const int64_t) strides_;

  TMOL_HD int64_t size(int i) const { return sizes_[i]; }
  TMOL_HD int64_t stride(int i) const { return strides_[i]; }

#if TMOL_PORTABLE_METAL
  TMOL_HD device T& operator[](int64_t i) const {
    return data_[strides_[0] * i];
  }
#else
  TMOL_HD T& operator[](int64_t i) const { return data_[strides_[0] * i]; }
#endif
};

}  // namespace portable
}  // namespace tmol
