#pragma once

// Plain parameter structs, free of the torch tensor layer.
//
// Split out of params.hh so that device-side scalar math can include the
// struct definitions without dragging in TensorUtil.h (and through it Eigen
// and torch), which exist only for the enable_tensor_view specializations.
// A Metal kernel receives these as bound buffers and needs the layout, not
// the tensor machinery. See tmol/tests/utility/portable/.

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
  his_taut_variant_both_ND1_and_NE2_protonated = 2,
  his_taut_variant_unresolved = 3,
};

}  // namespace compiled
}  // namespace details
}  // namespace io
}  // namespace tmol
