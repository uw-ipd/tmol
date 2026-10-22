// tmol::portable::Mat vs Eigen.
//
// The portable matrix replaced Eigen in tmol's device-side math. TView
// reinterprets torch storage as arrays of these types, so the in-memory
// element order and stride must match Eigen exactly -- those checks are
// byte-exact. Arithmetic is compared with a tolerance: reduction order and
// FP contraction differ from Eigen's expression templates, at the level of
// float rounding.
#include <cstdio>
#include <cstring>
#include <cmath>
#include <random>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <tmol/utility/portable/eigen_interop.hh>
using namespace tmol::portable;
static int bad = 0;
#define CLOSE(a, b)     \
  (std::fabs((a) - (b)) \
   <= 1e-12 * std::fmax(1.0, std::fmax(std::fabs(a), std::fabs(b))))
#define CHECK(c)                                                 \
  do {                                                           \
    if (!(c)) {                                                  \
      std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); \
      ++bad;                                                     \
    }                                                            \
  } while (0)

template <typename R, int Rr, int Cc>
void layout() {
  Mat<R, Rr, Cc> p{};
  Eigen::Matrix<R, Rr, Cc> e;
  int k = 0;
  for (int i = 0; i < Rr; ++i)
    for (int j = 0; j < Cc; ++j) {
      R val = R(++k);
      p(i, j) = val;
      e(i, j) = val;
    }
  // identical logical content must give identical bytes
  CHECK(std::memcmp(&p, e.data(), sizeof(p)) == 0);
}

int main() {
  layout<float, 3, 1>();
  layout<double, 3, 1>();
  layout<float, 4, 3>();
  layout<double, 4, 3>();
  layout<float, 3, 3>();
  layout<double, 3, 3>();
  layout<float, 4, 4>();
  layout<double, 4, 4>();

  std::mt19937 g(7);
  std::uniform_real_distribution<double> d(-2, 2);
  for (int t = 0; t < 5000; ++t) {
    Eigen::Matrix<double, 3, 3> eA, eB;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        eA(i, j) = d(g);
        eB(i, j) = d(g);
      }
    auto pA = from_eigen(eA), pB = from_eigen(eB);
    // transpose, row, col
    {
      auto pt = pA.transpose();
      Eigen::Matrix<double, 3, 3> et = eA.transpose();
      CHECK(std::memcmp(&pt, et.data(), sizeof(pt)) == 0);
    }
    for (int i = 0; i < 3; ++i) {
      auto pr = pA.row(i);
      Eigen::Matrix<double, 3, 1> er = eA.row(i).transpose();
      CHECK(std::memcmp(&pr, er.data(), sizeof(pr)) == 0);
      auto pc = pA.col(i);
      Eigen::Matrix<double, 3, 1> ec = eA.col(i);
      CHECK(std::memcmp(&pc, ec.data(), sizeof(pc)) == 0);
    }
    // matrix product
    auto pm = pA * pB;
    Eigen::Matrix<double, 3, 3> em = eA * eB;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) CHECK(CLOSE(pm(i, j), em(i, j)));
    // lvalue col proxy, as gen_coord.hh uses it
    Mat<double, 3, 3> pM{};
    Eigen::Matrix<double, 3, 3> eM;
    col_ref(pM, 0) = pA.col(0).normalized();
    eM.col(0) = eA.col(0).normalized();
    col_ref(pM, 1) = pA.col(1);
    eM.col(1) = eA.col(1);
    col_ref(pM, 2) = pM.col(0).cross(pM.col(1));
    eM.col(2) = eM.col(0).cross(eM.col(1));
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) CHECK(CLOSE(pM(i, j), eM(i, j)));
  }
  std::printf(bad ? "\n%d FAILURES\n" : "\nall Mat checks passed\n", bad);
  return bad != 0;
}
