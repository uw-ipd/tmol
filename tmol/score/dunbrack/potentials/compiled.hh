#pragma once

#include <Eigen/Core>
#include <tuple>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorCollection.h>
#include <tmol/utility/tensor/TensorPack.h>

#include <ATen/Tensor.h>

namespace tmol {
namespace score {
namespace rama {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <tmol::Device D, typename Real, typename Int>
struct DunbrackDispatch {
  static auto f(
      TCollection<Real, 2, D> rotameric_prob_tables,
      TCollection<Real, 2, D> rotameric_mean_tables,
      TCollection<Real, 2, D> rotameric_sdev_tables,
      TCollection<Real, 3, D> semirotameric_tables,

      // TView<Int, 1, D> rotameric_table_offsets,    // starting ind by table
      // set TView<Int, 1, D> semirot_table_offsets,      // starting ind by
      // table set

      TView<Eigen::Matrix<Real, 2, 1>, 1, D> coordinates,

      TView<Int, 1, D> ndihe_for_res,            // nres x 1
      TView<Int, 1, D> dihedral_offset_for_res,  // nres x 1
      TView<Int, 2, D> dihedral_inds,            // ndihe x 4 list of

      TView<Real, 1, D>
          dihedrals,  // Where the dihedrals will be written, ndihe x 1
      TView<Real, 2, D>
          dihedral_ddihe_dxyz,  // Where d dihe/ dxyz is stored, ndihe x 3
      TView<Real, 1, D>
          dihedral_dE_ddihe,  // Where d E/d dihe is stored, ndihe x 1

      TView<Real, 2, D> dihedral_dmean_ddihe,  // Where d chimean/d dbbdihe is
                                               // stored, nscdihe x 2
      TView<Real, 2, D> dihedral_dsdev_ddihe,  // Where d chisdev/d dbbdihe is
                                               // stored, nscdihe x 2

      TView<Int, 1, D>
          rotameric_assignment,  // Which rotamer for each residue, nres x 1

      TView<Int, 1, D> rotameric_table_set_offset,  // nres x 1
      TView<Int, 1, D> semirot_table_set_offset     // nres x 1
      ) -> TPack<Real, 1, D>;
};

}  // namespace potentials
}  // namespace rama
}  // namespace score
}  // namespace tmol
