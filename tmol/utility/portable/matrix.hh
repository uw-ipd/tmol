#pragma once

#include "math.hh"
#include "qualifiers.hh"

namespace tmol {
namespace portable {

namespace detail {
// Non-deduced context: lets `v * 2` convert the literal to Real instead of
// failing template argument deduction, matching Eigen's behavior.
template <typename T>
struct identity {
  typedef T type;
};
}  // namespace detail

template <typename T>
using identity_t = typename detail::identity<T>::type;

// Fixed-size matrix, layout-compatible with Eigen::Matrix<Real, R, C>.
//
// Storage is COLUMN-MAJOR, matching Eigen's default. tmol views torch storage
// as arrays of these types (see enable_tensor_view in TensorUtil.h), so the
// in-memory element order must not change as call sites migrate off Eigen.
//
// Deliberately an aggregate: no user-provided constructors, no base classes,
// no private members. That keeps it trivially copyable, preserves the
// `Vec<Real, 3> v{x, y, z}` and `Real3({0, 0, 0})` spellings the Eigen call
// sites use, and keeps the type usable from Metal, which has neither a
// standard library nor Eigen. Eigen conversions live in eigen_interop.hh so
// this header stays compilable by the Metal front end.
template <typename Real, int R, int C>
struct Mat {
  Real v[R * C];

  TMOL_HD TMOL_THREAD Real& operator()(int i, int j) { return v[i + j * R]; }
  TMOL_HD Real operator()(int i, int j) const { return v[i + j * R]; }

  // Vector-style access. Valid for any shape, but meaningful for R x 1.
  TMOL_HD TMOL_THREAD Real& operator[](int i) { return v[i]; }
  TMOL_HD Real operator[](int i) const { return v[i]; }

#if TMOL_PORTABLE_METAL
  // In MSL the address space is part of the implicit `this` pointer's type, so
  // the unqualified accessors above are thread-space-only and a Mat living in
  // a device buffer -- which every coordinate and derivative tensor element
  // does -- cannot be indexed through them. These overloads cover the other
  // address spaces tmol actually stores matrices in. They exist nowhere else;
  // on the host and under nvcc there is only one address space to name.
  TMOL_HD device Real& operator()(int i, int j) device {
    return v[i + j * R];
  }
  TMOL_HD Real operator()(int i, int j) const device { return v[i + j * R]; }
  TMOL_HD device Real& operator[](int i) device { return v[i]; }
  TMOL_HD Real operator[](int i) const device { return v[i]; }
  TMOL_HD device Real& operator()(int i) device { return v[i]; }
  TMOL_HD Real operator()(int i) const device { return v[i]; }

  TMOL_HD threadgroup Real& operator()(int i, int j) threadgroup {
    return v[i + j * R];
  }
  TMOL_HD Real operator()(int i, int j) const threadgroup {
    return v[i + j * R];
  }
  TMOL_HD threadgroup Real& operator[](int i) threadgroup { return v[i]; }
  TMOL_HD Real operator[](int i) const threadgroup { return v[i]; }
  TMOL_HD threadgroup Real& operator()(int i) threadgroup { return v[i]; }
  TMOL_HD Real operator()(int i) const threadgroup { return v[i]; }
#endif

  // Eigen spells single-index access with operator() as well as operator[].
  TMOL_HD TMOL_THREAD Real& operator()(int i) { return v[i]; }
  TMOL_HD Real operator()(int i) const { return v[i]; }

  TMOL_HD TMOL_THREAD Real* data() { return v; }
  TMOL_HD TMOL_THREAD Real const* data() const { return v; }

  TMOL_HD bool isZero() const {
    for (int i = 0; i < R * C; ++i) {
      if (v[i] != Real(0)) {
        return false;
      }
    }
    return true;
  }

  TMOL_HD static Mat Constant(Real value) {
    Mat r{};
    for (int i = 0; i < R * C; ++i) {
      r.v[i] = value;
    }
    return r;
  }

  TMOL_HD void reverseInPlace() {
    static_assert(C == 1, "reverseInPlace() requires an R x 1 vector");
    for (int i = 0; i < R / 2; ++i) {
      Real const t = v[i];
      v[i] = v[R - 1 - i];
      v[R - 1 - i] = t;
    }
  }

  TMOL_HD void setZero() {
    for (int i = 0; i < R * C; ++i) {
      v[i] = Real(0);
    }
  }

  TMOL_HD static Mat Zero() { return Mat{}; }

  TMOL_HD static Mat Identity() {
    Mat r{};
    int const n = (R < C) ? R : C;
    for (int i = 0; i < n; ++i) {
      r(i, i) = Real(1);
    }
    return r;
  }

  // Leading/trailing sub-vector, for R x 1 only; N is a compile-time size.
  template <int N>
  TMOL_HD Mat<Real, N, 1> head() const {
    static_assert(C == 1 && N <= R, "head<N>() requires an R x 1 vector");
    Mat<Real, N, 1> r{};
    for (int i = 0; i < N; ++i) {
      r.v[i] = v[i];
    }
    return r;
  }

  template <int N>
  TMOL_HD Mat<Real, N, 1> tail() const {
    static_assert(C == 1 && N <= R, "tail<N>() requires an R x 1 vector");
    Mat<Real, N, 1> r{};
    for (int i = 0; i < N; ++i) {
      r.v[i] = v[R - N + i];
    }
    return r;
  }

  TMOL_HD static int rows() { return R; }
  TMOL_HD static int cols() { return C; }
  TMOL_HD static int size() { return R * C; }

  TMOL_HD Real squaredNorm() const {
    Real acc = Real(0);
    for (int i = 0; i < R * C; ++i) {
      acc += v[i] * v[i];
    }
    return acc;
  }

  TMOL_HD Real norm() const { return math::sqrt(squaredNorm()); }

  TMOL_HD Real dot(TMOL_THREAD Mat const& o) const {
    Real acc = Real(0);
    for (int i = 0; i < R * C; ++i) {
      acc += v[i] * o.v[i];
    }
    return acc;
  }

  TMOL_HD Mat normalized() const {
    // Eigen leaves a zero vector unchanged rather than producing NaNs.
    Real const n2 = squaredNorm();
    if (n2 == Real(0)) {
      return *this;
    }
    Real const n = math::sqrt(n2);
    Mat r{};
    for (int i = 0; i < R * C; ++i) {
      r.v[i] = v[i] / n;
    }
    return r;
  }

  TMOL_HD Mat cross(TMOL_THREAD Mat const& o) const {
    static_assert(R == 3 && C == 1, "cross product requires a 3-vector");
    Mat r{};
    r.v[0] = v[1] * o.v[2] - v[2] * o.v[1];
    r.v[1] = v[2] * o.v[0] - v[0] * o.v[2];
    r.v[2] = v[0] * o.v[1] - v[1] * o.v[0];
    return r;
  }

  TMOL_HD Mat<Real, C, R> transpose() const {
    Mat<Real, C, R> r{};
    for (int i = 0; i < R; ++i) {
      for (int j = 0; j < C; ++j) {
        r(j, i) = (*this)(i, j);
      }
    }
    return r;
  }

  // const-qualified return: assigning to the result is a compile error, so a
  // missed `m.row(i) = v` cannot silently write to a temporary. Use row_ref().
  TMOL_HD const Mat<Real, C, 1> row(int i) const {
    Mat<Real, C, 1> r{};
    for (int j = 0; j < C; ++j) {
      r[j] = (*this)(i, j);
    }
    return r;
  }

  TMOL_HD const Mat<Real, R, 1> col(int j) const {
    Mat<Real, R, 1> r{};
    for (int i = 0; i < R; ++i) {
      r[i] = (*this)(i, j);
    }
    return r;
  }

  TMOL_HD TMOL_THREAD Mat& operator+=(TMOL_THREAD Mat const& o) {
    for (int i = 0; i < R * C; ++i) {
      v[i] += o.v[i];
    }
    return *this;
  }

  TMOL_HD TMOL_THREAD Mat& operator-=(TMOL_THREAD Mat const& o) {
    for (int i = 0; i < R * C; ++i) {
      v[i] -= o.v[i];
    }
    return *this;
  }

  TMOL_HD TMOL_THREAD Mat& operator*=(identity_t<Real> s) {
    for (int i = 0; i < R * C; ++i) {
      v[i] *= s;
    }
    return *this;
  }

  TMOL_HD TMOL_THREAD Mat& operator/=(identity_t<Real> s) {
    for (int i = 0; i < R * C; ++i) {
      v[i] /= s;
    }
    return *this;
  }
};

template <typename Real, int N>
using Vec = Mat<Real, N, 1>;

// Copies a fixed-size raw array into a vector. Replaces Eigen::Map at call
// sites that only read the mapped data.
template <typename Real, int N>
TMOL_HD Vec<Real, N> vec_from(TMOL_THREAD Real const* p) {
  Vec<Real, N> r{};
  for (int i = 0; i < N; ++i) {
    r.v[i] = p[i];
  }
  return r;
}

#define TMOL_MAT_BINOP(op)                                  \
  template <typename Real, int R, int C>                    \
  TMOL_HD Mat<Real, R, C> operator op(                      \
      TMOL_THREAD Mat<Real, R, C> const& a, TMOL_THREAD Mat<Real, R, C> const& b) { \
    Mat<Real, R, C> r{};                                    \
    for (int i = 0; i < R * C; ++i) {                       \
      r.v[i] = a.v[i] op b.v[i];                            \
    }                                                       \
    return r;                                               \
  }
TMOL_MAT_BINOP(+)
TMOL_MAT_BINOP(-)
#undef TMOL_MAT_BINOP

template <typename Real, int R, int C>
TMOL_HD Mat<Real, R, C> operator-(TMOL_THREAD Mat<Real, R, C> const& a) {
  Mat<Real, R, C> r{};
  for (int i = 0; i < R * C; ++i) {
    r.v[i] = -a.v[i];
  }
  return r;
}

template <typename Real, int R, int C>
TMOL_HD Mat<Real, R, C> operator*(
    TMOL_THREAD Mat<Real, R, C> const& a, identity_t<Real> s) {
  Mat<Real, R, C> r{};
  for (int i = 0; i < R * C; ++i) {
    r.v[i] = a.v[i] * s;
  }
  return r;
}

template <typename Real, int R, int C>
TMOL_HD Mat<Real, R, C> operator*(
    identity_t<Real> s, TMOL_THREAD Mat<Real, R, C> const& a) {
  return a * s;
}

template <typename Real, int R, int C>
TMOL_HD Mat<Real, R, C> operator/(
    TMOL_THREAD Mat<Real, R, C> const& a, identity_t<Real> s) {
  Mat<Real, R, C> r{};
  for (int i = 0; i < R * C; ++i) {
    r.v[i] = a.v[i] / s;
  }
  return r;
}

// Matrix product. Accumulation runs down the inner index in the same order
// Eigen's small-matrix product uses.
template <typename Real, int R, int K, int C>
TMOL_HD Mat<Real, R, C> operator*(
    TMOL_THREAD Mat<Real, R, K> const& a, TMOL_THREAD Mat<Real, K, C> const& b) {
  Mat<Real, R, C> r{};
  for (int i = 0; i < R; ++i) {
    for (int j = 0; j < C; ++j) {
      Real acc = Real(0);
      for (int k = 0; k < K; ++k) {
        acc += a(i, k) * b(k, j);
      }
      r(i, j) = acc;
    }
  }
  return r;
}

// Writable row/column views. Eigen returns assignable expressions from row()
// and col(); these proxies provide the same `m.col(0) = v;` spelling without
// expression templates.
template <typename Real, int R, int C>
struct RowRef {
  TMOL_THREAD Mat<Real, R, C>* m;
  int i;

  TMOL_HD TMOL_THREAD RowRef& operator=(TMOL_THREAD Mat<Real, C, 1> const& src) {
    for (int j = 0; j < C; ++j) {
      (*m)(i, j) = src[j];
    }
    return *this;
  }

  TMOL_HD TMOL_THREAD RowRef& operator+=(TMOL_THREAD Mat<Real, C, 1> const& src) {
    for (int j = 0; j < C; ++j) {
      (*m)(i, j) += src[j];
    }
    return *this;
  }

  TMOL_HD TMOL_THREAD RowRef& operator-=(TMOL_THREAD Mat<Real, C, 1> const& src) {
    for (int k = 0; k < C; ++k) {
      (*m)(i, k) -= src[k];
    }
    return *this;
  }

  TMOL_HD TMOL_THREAD RowRef& operator*=(identity_t<Real> s) {
    for (int k = 0; k < C; ++k) {
      (*m)(i, k) *= s;
    }
    return *this;
  }

  TMOL_HD TMOL_THREAD RowRef& operator/=(identity_t<Real> s) {
    for (int k = 0; k < C; ++k) {
      (*m)(i, k) /= s;
    }
    return *this;
  }

  TMOL_HD operator Mat<Real, C, 1>() const { return m->row(i); }
};

template <typename Real, int R, int C>
struct ColRef {
  TMOL_THREAD Mat<Real, R, C>* m;
  int j;

  TMOL_HD TMOL_THREAD ColRef& operator=(TMOL_THREAD Mat<Real, R, 1> const& src) {
    for (int i = 0; i < R; ++i) {
      (*m)(i, j) = src[i];
    }
    return *this;
  }

  TMOL_HD TMOL_THREAD ColRef& operator+=(TMOL_THREAD Mat<Real, R, 1> const& src) {
    for (int i = 0; i < R; ++i) {
      (*m)(i, j) += src[i];
    }
    return *this;
  }

  TMOL_HD TMOL_THREAD ColRef& operator-=(TMOL_THREAD Mat<Real, R, 1> const& src) {
    for (int k = 0; k < R; ++k) {
      (*m)(k, j) -= src[k];
    }
    return *this;
  }

  TMOL_HD TMOL_THREAD ColRef& operator*=(identity_t<Real> s) {
    for (int k = 0; k < R; ++k) {
      (*m)(k, j) *= s;
    }
    return *this;
  }

  TMOL_HD TMOL_THREAD ColRef& operator/=(identity_t<Real> s) {
    for (int k = 0; k < R; ++k) {
      (*m)(k, j) /= s;
    }
    return *this;
  }

  TMOL_HD operator Mat<Real, R, 1>() const { return m->col(j); }
};

template <typename Real, int R, int C>
TMOL_HD RowRef<Real, R, C> row_ref(TMOL_THREAD Mat<Real, R, C>& m, int i) {
  return RowRef<Real, R, C>{&m, i};
}

template <typename Real, int R, int C>
TMOL_HD ColRef<Real, R, C> col_ref(TMOL_THREAD Mat<Real, R, C>& m, int j) {
  return ColRef<Real, R, C>{&m, j};
}

}  // namespace portable
}  // namespace tmol
