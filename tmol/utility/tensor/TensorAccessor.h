#pragma once

#include <ATen/Tensor.h>
#include <torch/extension.h>

#include <stdint.h>
#include <algorithm>
#include <cstddef>
#include <iterator>

// Macros from #include <ATen/core/Macros.h>
#ifdef __CUDACC__
// Designates functions callable from the host (CPU) and the device (GPU)
#define AT_HOST_DEVICE __host__ __device__
#define AT_DEVICE __device__
#define AT_HOST __host__
#else
#define AT_HOST_DEVICE
#define AT_HOST
#define AT_DEVICE
#endif

#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

inline void handler(int sig, char* file, int line) {
  void* array[10];
  size_t size;

  // get void*'s for all entries on the stack
  size = backtrace(array, 10);

  // print out all the frames to stderr
  fprintf(stderr, "Error: signal %d from %s line %d:\n", sig, file, line);
  backtrace_symbols_fd(array, size, STDERR_FILENO);
  exit(1);
}

// Boundary assertions
#if !defined __CUDACC__
#define BOUNDARY_ASSERT(array_ptr, index) \
  if (index < 0) {                        \
    handler(1, __FILE__, __LINE__);       \
  }                                       \
  if (index >= array_ptr->sizes_[0]) {    \
    handler(1, __FILE__, __LINE__);       \
  }                                       \
                                          \
  assert(index >= 0);                     \
  assert(index < array_ptr->sizes_[0]);
#else
#define BOUNDARY_ASSERT(array_ptr, index)
#endif

namespace tmol {

enum class Device { CPU, CUDA };

enum class PtrTag { Restricted, Unrestricted };

// The PtrTraits argument to the TensorAccessor/TView
// is used to enable the __restrict__ keyword/modifier for the data
// passed to cuda.
template <typename T, PtrTag P>
struct PtrTraits {
  // Static assertion dependent on template parameter, only fails if template
  // is instantiated.
  static_assert(
      sizeof(T) < 0,
      "tmol::PtrTraits<T, P> with unknown typename T and PtrTag P.");
};

template <typename T>
struct PtrTraits<T, PtrTag::Unrestricted> {
  typedef T* PtrType;
};

template <typename T>
struct PtrTraits<T, PtrTag::Restricted> {
  typedef T* __restrict__ PtrType;
};

// TensorAccessorBase and TensorAccessor are used for both CPU and CUDA tensors.
// For CUDA tensors it is used in device code (only). This means that we
// restrict ourselves to functions and types available there (e.g. IntList
// isn't).

// The PtrTraits argument is only relevant to cuda to support `__restrict__`
// pointers.
template <typename T, size_t N, Device D, PtrTag P = PtrTag::Restricted>
class TensorAccessorBase {
 public:
  typedef typename PtrTraits<T, P>::PtrType PtrType;

  AT_HOST_DEVICE TensorAccessorBase()
      : data_(nullptr), sizes_(nullptr), strides_(nullptr) {}

  AT_HOST_DEVICE TensorAccessorBase(
      PtrType data_, const int64_t* sizes_, const int64_t* strides_)
      : data_(data_), sizes_(sizes_), strides_(strides_) {}

  AT_HOST_DEVICE int64_t stride(int64_t i) const { return strides_[i]; }
  AT_HOST_DEVICE int64_t size(int64_t i) const { return sizes_[i]; }
  AT_HOST_DEVICE const int64_t dim() const { return N; }
  AT_HOST_DEVICE T* data() { return data_; }
  AT_HOST_DEVICE const T* data() const { return data_; }

 protected:
  PtrType data_;
  const int64_t* sizes_;
  const int64_t* strides_;
};

// The `TensorAccessor` is typically instantiated for CPU `Tensor`s using
// `Tensor.accessor<T, N>()`.
// For CUDA `Tensor`s, `TView` is used on the host and only
// indexing on the device uses `TensorAccessor`s.
template <typename T, size_t N, Device D, PtrTag P = PtrTag::Restricted>
class TensorAccessor : public TensorAccessorBase<T, N, D, P> {
 public:
  typedef typename PtrTraits<T, P>::PtrType PtrType;

  AT_HOST_DEVICE TensorAccessor() : TensorAccessorBase<T, N, D, P>() {}

  AT_HOST_DEVICE TensorAccessor(
      PtrType data_, const int64_t* sizes_, const int64_t* strides_)
      : TensorAccessorBase<T, N, D, P>(data_, sizes_, strides_) {}

  AT_HOST_DEVICE TensorAccessor<T, N - 1, D, P> operator[](int64_t i) {
    BOUNDARY_ASSERT(this, i);
    return TensorAccessor<T, N - 1, D, P>(
        this->data_ + this->strides_[0] * i,
        this->sizes_ + 1,
        this->strides_ + 1);
  }

  AT_HOST_DEVICE const TensorAccessor<T, N - 1, D, P> operator[](
      int64_t i) const {
    BOUNDARY_ASSERT(this, i);
    return TensorAccessor<T, N - 1, D, P>(
        this->data_ + this->strides_[0] * i,
        this->sizes_ + 1,
        this->strides_ + 1);
  }
};

template <typename T, Device D, PtrTag P>
class TensorAccessor<T, 1, D, P> : public TensorAccessorBase<T, 1, D, P> {
 public:
  typedef typename PtrTraits<T, P>::PtrType PtrType;

  AT_HOST_DEVICE TensorAccessor() : TensorAccessorBase<T, 1, D, P>() {}

  AT_HOST_DEVICE TensorAccessor(
      PtrType data_, const int64_t* sizes_, const int64_t* strides_)
      : TensorAccessorBase<T, 1, D, P>(data_, sizes_, strides_) {}
  AT_HOST_DEVICE T& operator[](int64_t i) const {
    BOUNDARY_ASSERT(this, i);
    return this->data_[this->strides_[0] * i];
  }
};

// TViewBase and TView are used on for CUDA
// `Tensor`s on the host and as In contrast to `TensorAccessor`s, they copy the
// strides and sizes on instantiation (on the host) in order to transfer them on
// the device when calling kernels. On the device, indexing of multidimensional
// tensors gives to `TensorAccessor`s. Use RestrictPtrTraits as PtrTraits if you
// want the tensor's data pointer to be marked as __restrict__. Instantiation
// from data, sizes, strides is only needed on the host and std::copy isn't
// available on the device, so those functions are host only.
template <typename T, size_t N, Device D, PtrTag P = PtrTag::Restricted>
class TViewBase {
 public:
  typedef typename PtrTraits<T, P>::PtrType PtrType;

  AT_HOST_DEVICE TViewBase() : data_(NULL) {
#ifdef __CUDA_ARCH__
    for (int i = 0; i < N; ++i) {
      this->sizes_[i] = 0;
      this->strides_[i] = 0;
    }
#else
    std::fill(sizes_, sizes_ + N, 0);
    std::fill(strides_, strides_ + N, 0);
#endif
  }

  AT_HOST_DEVICE TViewBase(
      PtrType data_, const int64_t* sizes_, const int64_t* strides_)
      : data_(data_) {
#ifdef __CUDA_ARCH__
    for (int i = 0; i < N; ++i) {
      this->sizes_[i] = sizes_[i];
      this->strides_[i] = strides_[i];
    }
#else
    std::copy(sizes_, sizes_ + N, std::begin(this->sizes_));
    std::copy(strides_, strides_ + N, std::begin(this->strides_));
#endif
  }

  AT_HOST_DEVICE operator TensorAccessor<T, N, D, P>() const {
    return TensorAccessor<T, N, D, P>(
        this->data_, this->sizes_, this->strides_);
  }

  AT_HOST_DEVICE const int64_t dim() const { return N; }
  AT_HOST_DEVICE const int64_t& stride(int64_t i) const { return strides_[i]; }
  AT_HOST_DEVICE const int64_t& size(int64_t i) const { return sizes_[i]; }

  AT_HOST_DEVICE PtrType data() { return data_; }
  AT_HOST_DEVICE const PtrType data() const { return data_; }

 protected:
  PtrType data_;
  int64_t sizes_[N];
  int64_t strides_[N];
};

template <typename T, size_t N, Device D, PtrTag P = PtrTag::Restricted>
class TView : public TViewBase<T, N, D, P> {
 public:
  typedef typename PtrTraits<T, P>::PtrType PtrType;

  AT_HOST_DEVICE TView(
      PtrType data_, const int64_t* sizes_, const int64_t* strides_)
      : TViewBase<T, N, D, P>(data_, sizes_, strides_){};

  AT_HOST_DEVICE TView() : TViewBase<T, N, D, P>(){};

  AT_HOST_DEVICE TensorAccessor<T, N - 1, D, P> operator[](int64_t i) {
    BOUNDARY_ASSERT(this, i);
    int64_t* new_sizes = this->sizes_ + 1;
    int64_t* new_strides = this->strides_ + 1;
    return TensorAccessor<T, N - 1, D, P>(
        this->data_ + this->strides_[0] * i, new_sizes, new_strides);
  }

  AT_HOST_DEVICE const TensorAccessor<T, N - 1, D, P> operator[](
      int64_t i) const {
    BOUNDARY_ASSERT(this, i);
    const int64_t* new_sizes = this->sizes_ + 1;
    const int64_t* new_strides = this->strides_ + 1;
    return TensorAccessor<T, N - 1, D, P>(
        this->data_ + this->strides_[0] * i, new_sizes, new_strides);
  }

  AT_HOST_DEVICE
  TView<T, N - 1, D, P> slice_one(int dim, int val) const {
    int64_t new_sizes[N - 1];
    int64_t new_strides[N - 1];
    int count = 0;
    for (int i = 0; i < N; ++i) {
      if (i != dim) {
        new_sizes[count] = this->sizes_[i];
        new_strides[count] = this->strides_[i];
        ++count;
      }
    }
    return TView<T, N - 1, D, P>(
        this->data_ + this->strides_[dim] * val, new_sizes, new_strides);
  }
};

template <typename T, Device D, PtrTag P>
class TView<T, 1, D, P> : public TViewBase<T, 1, D, P> {
 public:
  typedef typename PtrTraits<T, P>::PtrType PtrType;

  AT_HOST_DEVICE TView(
      PtrType data_, const int64_t* sizes_, const int64_t* strides_)
      : TViewBase<T, 1, D, P>(data_, sizes_, strides_){};

  AT_HOST_DEVICE TView() : TViewBase<T, 1, D, P>(){};

  AT_HOST_DEVICE T& operator[](int64_t i) {
    BOUNDARY_ASSERT(this, i);
    return this->data_[this->strides_[0] * i];
  }
  AT_HOST_DEVICE T& operator[](int64_t i) const {
    BOUNDARY_ASSERT(this, i);
    return this->data_[this->strides_[0] * i];
  }
};

}  // namespace tmol
