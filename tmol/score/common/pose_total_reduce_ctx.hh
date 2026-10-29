#pragma once

// State the pose-total reduction reads.
//
// Templated on the view so the field list is declared once; see elec_ctx.hh.

#include <tmol/utility/portable/device.hh>
#include <tmol/utility/portable/matrix.hh>

namespace tmol {
namespace score {
namespace common {

template <template <typename, int> class View, typename Real>
struct PoseTotalReduceCtxT {
  // [n_terms, n_poses, max_n_blocks, max_n_blocks] -- one cell per block pair
  View<Real, 4> cells;
  // [n_terms, n_poses, 1, 1]
  View<Real, 4> totals;

  template <typename F>
  TMOL_HD void visit_views(F f) {
    f(cells);
    f(totals);
  }

  enum { n_views = 2 };
  // Launch scalars in the order the host writes and the shader reads.
  template <typename F>
  TMOL_HD void visit_scalars(F f) {
    f(n_cells);
  }

  // max_n_blocks * max_n_blocks, the number of cells per (term, pose)
  int n_cells = {};
};

}  // namespace common
}  // namespace score
}  // namespace tmol
