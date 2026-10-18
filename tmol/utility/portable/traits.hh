#pragma once

#include "qualifiers.hh"

// Minimal type traits for device-side code.
//
// <type_traits> is unavailable in a Metal translation unit -- libc++ itself
// does not compile as MSL -- so the handful of traits tmol's device headers
// use are spelled here. Only what tmol actually needs is provided.

namespace tmol {
namespace portable {

template <bool B, typename T = void>
struct enable_if {};

template <typename T>
struct enable_if<true, T> {
  typedef T type;
};

template <bool B, typename T = void>
using enable_if_t = typename enable_if<B, T>::type;

template <typename A, typename B>
struct is_same {
  // Metal does not support derived classes, so these cannot inherit from an
  // integral_constant; the value is an enumerator rather than a static data
  // member because program-scope variables in MSL must be in the constant
  // address space.
  enum { value = 0 };
};

template <typename A>
struct is_same<A, A> {
  enum { value = 1 };
};

template <typename T>
struct is_integral {
  enum { value = 0 };
};

template <typename T>
struct is_floating_point {
  enum { value = 0 };
};

#define TMOL_PORTABLE_TRAIT_TRUE(TRAIT, TYPE) \
  template <>                                 \
  struct TRAIT<TYPE> {                        \
    enum { value = 1 };                       \
  }

TMOL_PORTABLE_TRAIT_TRUE(is_integral, bool);
TMOL_PORTABLE_TRAIT_TRUE(is_integral, char);
TMOL_PORTABLE_TRAIT_TRUE(is_integral, signed char);
TMOL_PORTABLE_TRAIT_TRUE(is_integral, unsigned char);
TMOL_PORTABLE_TRAIT_TRUE(is_integral, short);
TMOL_PORTABLE_TRAIT_TRUE(is_integral, unsigned short);
TMOL_PORTABLE_TRAIT_TRUE(is_integral, int);
TMOL_PORTABLE_TRAIT_TRUE(is_integral, unsigned int);
TMOL_PORTABLE_TRAIT_TRUE(is_integral, long);
TMOL_PORTABLE_TRAIT_TRUE(is_integral, unsigned long);
#if !TMOL_PORTABLE_METAL
// Metal rejects the `long long` spelling outright; there, `long` is already
// the 64-bit type and the specializations above cover it.
TMOL_PORTABLE_TRAIT_TRUE(is_integral, long long);
TMOL_PORTABLE_TRAIT_TRUE(is_integral, unsigned long long);
#endif

TMOL_PORTABLE_TRAIT_TRUE(is_floating_point, float);
#if !TMOL_PORTABLE_METAL
// Metal has no double; naming the type at all is an error there.
TMOL_PORTABLE_TRAIT_TRUE(is_floating_point, double);
#endif

#undef TMOL_PORTABLE_TRAIT_TRUE

template <typename T>
struct is_arithmetic {
  enum { value = is_integral<T>::value || is_floating_point<T>::value };
};

template <typename T>
struct remove_reference {
  typedef T type;
};

template <typename T>
struct remove_reference<TMOL_THREAD T&> {
  typedef T type;
};

template <typename T>
using remove_reference_t = typename remove_reference<T>::type;

}  // namespace portable
}  // namespace tmol
