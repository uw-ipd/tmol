#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <tuple>

#include <cppitertools/product.hpp>
#include <cppitertools/range.hpp>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>

#include "dispatch.hh"

namespace tmol {
namespace score {
namespace common {

using iter::product;
using iter::range;
using std::tie;
using std::tuple;
using tmol::new_tensor;
using tmol::TView;

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <>
struct ExhaustiveDispatch<tmol::Device::CPU> {
  static const tmol::Device D = tmol::Device::CPU;

  ExhaustiveDispatch(int n_i, int n_j) : n_i(n_i), n_j(n_j) {}

  template <typename Real>
  int scan(
      Real threshold_distance,
      TView<Vec<Real, 3>, 1, D> coords_i,
      TView<Vec<Real, 3>, 1, D> coords_j) {
    return n_i * n_j;
  }

  template <typename funct_t>
  void score(funct_t f) {
    int oind = 0;
    for (int i = 0; i < n_i; i++) {
      for (int j = 0; j < n_j; j++) {
        f(oind, i, j);
        oind++;
      }
    }
  }

  int n_i, n_j;
};

template <>
struct ExhaustiveTriuDispatch<tmol::Device::CPU> {
  static const tmol::Device D = tmol::Device::CPU;

  ExhaustiveTriuDispatch(int n_i, int n_j) : n_i(n_i), n_j(n_j) {}

  template <typename Real>
  int scan(
      Real threshold_distance,
      TView<Vec<Real, 3>, 1, D> coords_i,
      TView<Vec<Real, 3>, 1, D> coords_j) {
    int n_hit = 0;

    for (int i = 0; i < n_i; i++) {
      n_hit += n_j - i;
    }

    return n_hit;
  }

  template <typename funct_t>
  void score(funct_t f) {
    int oind = 0;
    for (int i = 0; i < n_i; i++) {
      for (int j = i; j < n_j; j++) {
        f(oind, i, j);
        oind++;
      }
    }
  }

  int n_i, n_j;
};

template <>
struct NaiveDispatch<tmol::Device::CPU> {
  static const tmol::Device D = tmol::Device::CPU;

  NaiveDispatch(int n_i, int n_j) : n_i(n_i), n_j(n_j), n_ind(0) {
    tie(inds_t, inds) = new_tensor<int, 2, D>({n_i * n_j, 2});
  }

  template <typename Real>
  int scan(
      Real threshold_distance,
      TView<Vec<Real, 3>, 1, D> coords_i,
      TView<Vec<Real, 3>, 1, D> coords_j) {
    const Eigen::AlignedBox<Real, 3> tbox(
        Vec<Real, 3>(
            -threshold_distance, -threshold_distance, -threshold_distance),
        Vec<Real, 3>(
            threshold_distance, threshold_distance, threshold_distance));

    n_ind = 0;

    for (int i = 0; i < n_i; ++i) {
      for (int j = 0; j < n_j; ++j) {
        if (tbox.contains(coords_i[i] - coords_j[j])) {
          inds[n_ind][0] = i;
          inds[n_ind][1] = j;
          n_ind++;
        }
      }
    }

    return n_ind;
  }

  template <typename funct_t>
  void score(funct_t f) {
    for (int o = 0; o < n_ind; o++) {
      f(o, inds[o][0], inds[o][1]);
    }
  }

  int n_i;
  int n_j;

  int n_ind;
  at::Tensor inds_t;
  TView<int, 2, D> inds;
};

template <>
struct NaiveTriuDispatch<tmol::Device::CPU> {
  static const tmol::Device D = tmol::Device::CPU;

  NaiveTriuDispatch(int n_i, int n_j) : n_i(n_i), n_j(n_j), n_ind(0) {
    tie(inds_t, inds) = new_tensor<int, 2, D>({n_i * n_j, 2});
  }

  template <typename Real>
  int scan(
      Real threshold_distance,
      TView<Vec<Real, 3>, 1, D> coords_i,
      TView<Vec<Real, 3>, 1, D> coords_j) {
    const Eigen::AlignedBox<Real, 3> tbox(
        Vec<Real, 3>(
            -threshold_distance, -threshold_distance, -threshold_distance),
        Vec<Real, 3>(
            threshold_distance, threshold_distance, threshold_distance));

    n_ind = 0;

    for (int i = 0; i < n_i; ++i) {
      for (int j = i; j < n_j; ++j) {
        if (tbox.contains(coords_i[i] - coords_j[j])) {
          inds[n_ind][0] = i;
          inds[n_ind][1] = j;
          n_ind++;
        }
      }
    }

    return n_ind;
  }

  template <typename funct_t>
  void score(funct_t f) {
    for (int o = 0; o < n_ind; o++) {
      f(o, inds[o][0], inds[o][1]);
    }
  }

  int n_i;
  int n_j;

  int n_ind;
  at::Tensor inds_t;
  TView<int, 2, D> inds;
};

template <>
struct ExhaustiveOMPDispatch<tmol::Device::CPU> {
  static const tmol::Device D = tmol::Device::CPU;

  ExhaustiveOMPDispatch(int n_i, int n_j) : n_i(n_i), n_j(n_j) {}

  template <typename Real>
  int scan(Real threshold_distance, TView<Vec<Real, 3>, 1, D> coords_i,
           TView<Vec<Real, 3>, 1, D> coords_j) {
    return n_i * n_j;
  }

  template <typename funct_t>
  void score(funct_t f) {
#pragma omp parallel for
    for (int i = 0; i < n_i; i++) {
      for (int j = 0; j < n_j; j++) {
        auto oind = j + i * n_j;
        f(oind, i, j);
      }
    }
  }

  int n_i, n_j;
};

template <>
struct AABBDispatch<tmol::Device::CPU> {
  static const tmol::Device D = tmol::Device::CPU;
  static const int N_BBOX = 8;
  // 'Real' should be template arg for whole struct?
  typedef Eigen::AlignedBox<double, 3> BBox;

  AABBDispatch(int n_i, int n_j) : n_i(n_i), n_j(n_j) {
  }

  template<typename Real>
  void make_bboxes(
    TView<Vec<Real, 3>, 1, D> coords_i,
    TView<Vec<Real, 3>, 1, D> coords_j
  ){
    int n_bb_i = std::ceil(n_i/N_BBOX);
    int n_bb_j = std::ceil(n_j/N_BBOX);
    tie(bbox_i_t, bbox_i) = new_tensor<BBox, 1, D>(n_bb_i);
    tie(bbox_j_t, bbox_j) = new_tensor<BBox, 1, D>(n_bb_j);

    #pragma omp parallel for
    for(int i = 0; i < n_i; ++i){
      if(i%N_BBOX==0)
        bbox_i[i/N_BBOX] = BBox(coords_i[i], coords_i[i]);
      else
        bbox_i[i/N_BBOX].extend(coords_i[i]);
    }

    #pragma omp parallel for
    for(int j = 0; j < n_j; ++j){
      if(j%N_BBOX==0)
        bbox_j[j/N_BBOX] = BBox(coords_j[j], coords_j[j]);
      else
        bbox_j[j/N_BBOX].extend(coords_j[j]);
    }

  }

  template <typename Real>
  int find_pairs(Real threshold_distance){
    tie(pairs_t, pairs) = new_tensor<int, 2, D>({bbox_i.size(0) * bbox_j.size(0), 2});
    int ipair = -1;
    #pragma omp parallel for
    for(int ibb = 0; ibb < bbox_i.size(0); ++ibb){
      for(int jbb = 0; jbb < bbox_j.size(0); ++jbb){
        auto dis = bbox_i[ibb].exteriorDistance(bbox_j[jbb]);
        if(dis <= threshold_distance){
          #pragma omp atomic
          ipair++;
          pairs[ipair][0] = ibb;
          pairs[ipair][1] = jbb;
        }
      }
    }
    return ipair + 1;
  }

  template <typename Real>
  int scan(Real threshold_distance,
           TView<Vec<Real, 3>, 1, D> coords_i,
           TView<Vec<Real, 3>, 1, D> coords_j)
  {
    make_bboxes(coords_i, coords_j);
    n_pairs = find_pairs(threshold_distance + 0.1);
    return n_pairs * N_BBOX * N_BBOX;
  }

  template <typename funct_t>
  void score(funct_t f) {
    #pragma omp parallel for
    for(int ipair = 0; ipair < n_pairs; ++ipair){
      int ibb = pairs[ipair][0];
      int jbb = pairs[ipair][1];
      for(int iofst = 0; iofst < N_BBOX; ++iofst){
        int i = N_BBOX * ibb + iofst;
        if(i >= n_i) break;
        for(int jofst = 0; jofst < N_BBOX; ++jofst){
          int j = N_BBOX * jbb + jofst;
          if(j >= n_j) break;
          int oind = ipair * N_BBOX * N_BBOX + iofst * N_BBOX + jofst;
          f(oind, i, j);
        }
      }
    }
  }

  int n_i, n_j, n_pairs;
  at::Tensor bbox_i_t, bbox_j_t, pairs_t;
  TView<BBox, 1, D> bbox_i, bbox_j;
  TView<int, 2, D> pairs;


};


}  // namespace common
}  // namespace score
}  // namespace tmol
