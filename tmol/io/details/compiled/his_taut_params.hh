#pragma once

namespace tmol {
namespace io {
namespace details {
namespace compiled {

template <typename Int>
struct HisAtomIndsInCanonicalOrdering {
  Int his_ND1_in_co;
  Int his_NE2_in_co;
  Int his_HD1_in_co;
  Int his_HE2_in_co;
  Int his_HN_in_co;
  Int his_NH_in_co;
  Int his_NN_in_co;
  Int his_CG_in_co;
};

// What decision have we made for a particular
// histidine about its tautomerization state?
// This enum must match the ordering of entries in
// HisTautomerResolution from the file
// tmol/io/details/his_taut_resolution.py
enum HisTautomerResolution {
  his_taut_missing_atoms = 0,
  his_taut_HD1,
  his_taut_HE2,
  his_taut_NH_is_ND1,
  his_taut_NN_is_ND1,
  his_taut_HD1_HE2,
  his_taut_unresolved,
};

enum HisTautVariants {
  his_taut_variant_NE2_protonated = 0,
  his_taut_variant_ND1_protonated = 1,
};

}  // namespace compiled
}  // namespace details
}  // namespace io
}  // namespace tmol

namespace tmol {

template <typename Int>
struct enable_tensor_view<
    io::details::compiled::HisAtomIndsInCanonicalOrdering<Int>> {
  static const bool enabled = enable_tensor_view<Int>::enabled;
  static const at::ScalarType scalar_type() {
    return enable_tensor_view<Int>::scalar_type();
  }
  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0)
               ? sizeof(
                     io::details::compiled::HisAtomIndsInCanonicalOrdering<Int>)
                     / sizeof(Int)
               : 0;
  }

  typedef typename enable_tensor_view<Int>::PrimitiveType PrimitiveType;
};

}  // namespace tmol
