#pragma once

#include <tuple>

#include <Eigen/Core>

#include <tmol/score/hbond/potentials/params.hh>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/context_manager.hh>

namespace tmol {
namespace score {
namespace gridscore {
namespace potentials {

template <typename Real, int N>
using HBVec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
struct LiveHBondDispatch {
  static auto forward(
      ContextManager& mgr,
      TView<HBVec<Real, 3>, 2, D> ligand_coords,
      TView<Int, 2, D> ligand_donor_atoms,
      TView<Int, 1, D> ligand_donor_types,
      TView<Int, 2, D> ligand_acceptor_atoms,
      TView<Int, 1, D> ligand_acceptor_types,
      TView<HBVec<Real, 3>, 2, D> receptor_donors,
      TView<Int, 1, D> receptor_donor_types,
      TView<HBVec<Real, 3>, 2, D> receptor_acceptors,
      TView<Int, 1, D> receptor_acceptor_types,
      TView<Real, 1, D> cell_origin,
      TView<Int, 1, D> cell_dims,
      TView<Int, 1, D> receptor_donor_cell_offsets,
      TView<Int, 1, D> receptor_donor_cell_indices,
      TView<Int, 1, D> receptor_acceptor_cell_offsets,
      TView<Int, 1, D> receptor_acceptor_cell_indices,
      TView<hbond::potentials::HBondPairParams<Real>, 2, D> pair_params,
      TView<hbond::potentials::HBondPolynomials<double>, 2, D>
          pair_polynomials,
      TView<hbond::potentials::HBondGlobalParams<Real>, 1, D> global_params)
      -> std::tuple<TPack<Real, 1, D>, TPack<HBVec<Real, 3>, 2, D>>;
};

}  // namespace potentials
}  // namespace gridscore
}  // namespace score
}  // namespace tmol
