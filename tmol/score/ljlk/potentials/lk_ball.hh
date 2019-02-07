#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

#include <tmol/score/common/geom.hh>
#include "lk_isotropic.hh"

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

template <typename Real>
struct build_don_water {
  typedef Eigen::Matrix<Real, 3, 1> Real3;
  typedef Eigen::Matrix<Real, 3, 3> RealMat;

  struct dV_t {
    RealMat dD;
    RealMat dH;

    def astuple() { return make_tuple(dD, dH); }

    static def Zero()->dV_t { return {RealMat::Zero(), RealMat::Zero()}; }
  };

  static def V(Real3 D, Real3 H, Real dist)->Real3 {
    return D + dist * (H - D).normalized();
  }

  static def dV(Real3 D, Real3 H, Real dist)->dV_t {
    Real dhx = -D[0] + H[0];
    Real dhx2 = dhx * dhx;
    Real dhy = -D[1] + H[1];
    Real dhy2 = dhy * dhy;
    Real dhz = -D[2] + H[2];
    Real dhz2 = dhz * dhz;
    Real dh2 = dhx2 + dhy2 + dhz2;
    Real dist_norm = dist / std::sqrt(dh2);
    Real dist_norm_deriv = dist_norm / dh2;

    Eigen::Matrix<Real, 3, 3> dW_dD;
    dW_dD(0, 0) = dhx2 * dist_norm / dh2 + (1 - dist_norm);
    dW_dD(0, 1) = dhy * dhx * dist_norm_deriv;
    dW_dD(0, 2) = dhz * dhx * dist_norm_deriv;
    dW_dD(1, 0) = dhy * dhx * dist_norm_deriv;
    dW_dD(1, 1) = dhy2 * dist_norm_deriv + (1 - dist_norm);
    dW_dD(1, 2) = dhz * dhy * dist_norm_deriv;
    dW_dD(2, 0) = dhz * dhx * dist_norm_deriv;
    dW_dD(2, 1) = dhz * dhy * dist_norm_deriv;
    dW_dD(2, 2) = dhz2 * dist_norm_deriv + (1 - dist_norm);

    Eigen::Matrix<Real, 3, 3> dW_dH;
    dW_dH(0, 0) = dist_norm - dhx * dist_norm_deriv * dhx;
    dW_dH(0, 1) = -dhy * dist_norm_deriv * dhx;
    dW_dH(0, 2) = -dhx * dhz * dist_norm_deriv;
    dW_dH(1, 0) = -dhx * dist_norm_deriv * dhy;
    dW_dH(1, 1) = dist_norm - dhy * dist_norm_deriv * dhy;
    dW_dH(1, 2) = -dhz * dist_norm_deriv * dhy;
    dW_dH(2, 0) = -dhx * dist_norm_deriv * dhz;
    dW_dH(2, 1) = -dhy * dist_norm_deriv * dhz;
    dW_dH(2, 2) = dist_norm - dhz * dist_norm_deriv * dhz;

    return {dW_dD, dW_dH};
  }
};

template <typename Real>
struct build_acc_water {
  typedef Eigen::Matrix<Real, 3, 1> Real3;
  typedef Eigen::Matrix<Real, 3, 3> RealMat;

  struct dV_t {
    RealMat dA;
    RealMat dB;
    RealMat dB0;

    def astuple() { return make_tuple(dA, dB, dB0); }

    static def Zero()->dV_t {
      return {RealMat::Zero(), RealMat::Zero(), RealMat::Zero()};
    }
  };

  static def V(Real3 A, Real3 B, Real3 B0, Real dist, Real angle, Real torsion)
      ->Real3 {
    const Real pi = EIGEN_PI;

    // Generate orientation frame
    Eigen::Matrix<Real, 3, 3> M;
    M.col(0) = (A - B).normalized();
    M.col(1) = (B0 - B);
    M.col(2) = M.col(0).cross(M.col(1));

    Real M2_norm = M.col(2).norm();
    if (M2_norm == 0) {
      // if a/b/c collinear, set M[:,2] to an arbitrary vector perp to
      // M[:,0]
      if (M(0, 0) != 1) {
        M.col(1) = Real3({1, 0, 0});
        M.col(2) = M.col(0).cross(M.col(1));
      } else {
        M.col(1) = Real3({0, 1, 0});
        M.col(2) = M.col(0).cross(M.col(1));
      }
      M2_norm = M.col(2).norm();
    }
    M.col(2) /= M2_norm;
    M.col(1) = M.col(2).cross(M.col(0));

    // Build water in frame
    return (
        M
            * Real3({dist * std::cos(pi - angle),
                     dist * std::sin(pi - angle) * std::cos(torsion),
                     dist * std::sin(pi - angle) * std::sin(torsion)})
        + A);
  }

  static def dV(Real3 A, Real3 B, Real3 B0, Real dist, Real angle, Real torsion)
      ->dV_t {
    const Real pi = EIGEN_PI;

    // clang-format off
    Real x101 = -B[2];
    Real x102 = A[2] + x101;
    Real x103 = -B[0];
    Real x104 = B0[0] + x103;
    Real x105 = A[0] + x103;
    Real x106 = x105 * x105;
    Real x107 = -B[1];
    Real x108 = A[1] + x107;
    Real x109 = x108 * x108;
    Real x110 = x102 * x102;
    Real x111 = x106 + x109 + x110;
    Real x112 = 1 / std::sqrt(x111);
    Real x113 = x104 * x112;
    Real x114 = B0[2] + x101;
    Real x115 = x112 * x114;
    Real x116 = x102 * x113 - x105 * x115;
    Real x117 = B0[1] + x107;
    Real x118 = x112 * x117;
    Real x119 = x105 * x118 - x108 * x113;
    Real x120 = -x102 * x118 + x108 * x115;
    Real x121 = x116 * x116 + x119 * x119 + x120 * x120;
    Real x122 = 1 / std::sqrt(x121);
    Real x123 = std::pow(x111, (-1.5));
    Real x124 = x123 * (-A[0] + B[0]);
    Real x125 = x122 * x124;
    Real x126 = x102 * x125;
    Real x127 = x108 * x125;
    Real x128 = x108 * x112;
    Real x129 = x105 * x124;
    Real x130 = x117 * x129;
    Real x131 = x104 * x124;
    Real x132 = x108 * x131;
    Real x133 = x122 * (x118 + x130 - x132);
    Real x134 = -x115;
    Real x135 = x102 * x131;
    Real x136 = x114 * x129;
    Real x137 = x134 + x135 - x136;
    Real x138 = x102 * x112;
    Real x139 = x122 * x138;
    Real x140 = x116 * x138;
    Real x141 = std::pow(x121, (-1.5));
    Real x142 = x108 * x114;
    Real x143 = x124 * x142;
    Real x144 = x102 * x117;
    Real x145 = x124 * x144;
    Real x146 = x120 / 2;
    Real x147 = 2 * x118;
    Real x148 = x119 / 2;
    Real x149 = 2 * x115;
    Real x150 = -x149;
    Real x151 = x116 / 2;
    Real x152 = x141 * (
        -x146 * (2 * x143 - 2 * x145)
        - x148 * (2 * x130 - 2 * x132 + x147)
        - x151 * (2 * x135 - 2 * x136 + x150)
    );
    Real x153 = x119 * x128;
    Real x154 = torsion;
    Real x155 = pi - angle;
    Real x156 = dist * std::sin(x155);
    Real x157 = x156 * std::cos(x154);
    Real x158 = dist * std::cos(x155);
    Real x159 = x143 - x145;
    Real x160 = x156 * std::sin(x154);
    Real x161 = x122 * x160;
    Real x162 = x152 * x160;
    Real x163 = x112 * x158;
    Real x164 = x163 + 1;
    Real x165 = x124 * x158;
    Real x166 = x112 * x122;
    Real x167 = x119 * x166;
    Real x168 = x122 * x129;
    Real x169 = x105 * x112;
    Real x170 = x119 * x169;
    Real x171 = x120 * x138;
    Real x172 = x116 * x166;
    Real x173 = -x172;
    Real x174 = x122 * x128;
    Real x175 = x122 * x169;
    Real x176 = x120 * x128;
    Real x177 = x116 * x169;
    Real x178 = x123 * (-A[1] + B[1]);
    Real x179 = x105 * x178;
    Real x180 = x142 * x178;
    Real x181 = x144 * x178;
    Real x182 = x115 + x180 - x181;
    Real x183 = x114 * x179;
    Real x184 = x104 * x178;
    Real x185 = x102 * x184;
    Real x186 = 2 * x113;
    Real x187 = -x186;
    Real x188 = x117 * x179;
    Real x189 = x108 * x184;
    Real x190 = x141 * (
        -x146 * (x149 + 2 * x180 - 2 * x181)
        - x148 * (x187 + 2 * x188 - 2 * x189)
        - x151 * (-2 * x183 + 2 * x185)
    );
    Real x191 = x160 * x190;
    Real x192 = -x167;
    Real x193 = x122 * x178;
    Real x194 = x102 * x193;
    Real x195 = x108 * x193;
    Real x196 = -x183 + x185;
    Real x197 = -x113;
    Real x198 = x188 - x189 + x197;
    Real x199 = x122 * x179;
    Real x200 = x158 * x178;
    Real x201 = x120 * x166;
    Real x202 = x123 * (-A[2] + B[2]);
    Real x203 = x105 * x202;
    Real x204 = -x118;
    Real x205 = x142 * x202;
    Real x206 = x144 * x202;
    Real x207 = x204 + x205 - x206;
    Real x208 = x117 * x203;
    Real x209 = x104 * x202;
    Real x210 = x108 * x209;
    Real x211 = x114 * x203;
    Real x212 = x102 * x209;
    Real x213 = -x147;
    Real x214 = x141 * (
        -x146 * (2 * x205 - 2 * x206 + x213)
        - x148 * (2 * x208 - 2 * x210)
        - x151 * (x186 - 2 * x211 + 2 * x212)
    );
    Real x215 = x160 * x214;
    Real x216 = x122 * x202;
    Real x217 = x102 * x216;
    Real x218 = x108 * x216;
    Real x219 = x208 - x210;
    Real x220 = x113 - x211 + x212;
    Real x221 = x158 * x202;
    Real x222 = -x201;
    Real x223 = x122 * x203;
    Real x224 = -x163;
    Real x225 = x106 * x123;
    Real x226 = x105 * x123;
    Real x227 = x142 * x226;
    Real x228 = x144 * x226;
    Real x229 = x227 - x228;
    Real x230 = 2 * x227;
    Real x231 = 2 * x228;
    Real x232 = 2 * x128;
    Real x233 = x117 * x225;
    Real x234 = x104 * x226;
    Real x235 = x108 * x234;
    Real x236 = 2 * x138;
    Real x237 = x114 * x225;
    Real x238 = x102 * x234;
    Real x239 = x141 * (
        -x146 * (x230 - x231)
        - x148 * (x213 + x232 + 2 * x233 - 2 * x235)
        - x151 * (x149 - x236 - 2 * x237 + 2 * x238)
    );
    Real x240 = x160 * x239;
    Real x241 = x102 * x226;
    Real x242 = x122 * x241;
    Real x243 = x116 * x242;
    Real x244 = x108 * x226;
    Real x245 = x122 * x244;
    Real x246 = x119 * x245;
    Real x247 = x115 - x138 - x237 + x238;
    Real x248 = x128 + x204 + x233 - x235;
    Real x249 = x158 * x244;
    Real x250 = x122 * x225;
    Real x251 = x158 * x241;
    Real x252 = x109 * x123;
    Real x253 = x114 * x252;
    Real x254 = x108 * x123;
    Real x255 = x144 * x254;
    Real x256 = x134 + x138 + x253 - x255;
    Real x257 = x102 * x254;
    Real x258 = x104 * x257;
    Real x259 = 2 * x258;
    Real x260 = 2 * x169;
    Real x261 = x104 * x252;
    Real x262 = x117 * x244;
    Real x263 = x141 * (
        -x146 * (x150 + x236 + 2 * x253 - 2 * x255)
        - x148 * (x186 - x260 - 2 * x261 + 2 * x262)
        - x151 * (-x230 + x259)
    );
    Real x264 = x160 * x263;
    Real x265 = x122 * x252;
    Real x266 = x122 * x257;
    Real x267 = -x227 + x258;
    Real x268 = x113 - x169 - x261 + x262;
    Real x269 = x120 * x266;
    Real x270 = x158 * x257;
    Real x271 = x110 * x123;
    Real x272 = x117 * x271;
    Real x273 = x102 * x123 * x142;
    Real x274 = x118 - x128 - x272 + x273;
    Real x275 = x104 * x271;
    Real x276 = x114 * x241;
    Real x277 = x141 * (
        -x146 * (x147 - x232 - 2 * x272 + 2 * x273)
        - x148 * (x231 - x259)
        - x151 * (x187 + x260 + 2 * x275 - 2 * x276)
    );
    Real x278 = x160 * x277;
    Real x279 = x122 * x271;
    Real x280 = x228 - x258;
    Real x281 = x169 + x197 + x275 - x276;
    Real x282 = x141 * (-x140 + x153);
    Real x283 = x160 * x282;
    Real x284 = x122 / x111;
    Real x285 = x109 * x284;
    Real x286 = x110 * x284;
    Real x287 = x139 * x160;
    Real x288 = x105 * x284;
    Real x289 = -x108 * x288;
    Real x290 = x160 * x174;
    Real x291 = -x102 * x288;
    Real x292 = x141 * (-x170 + x171);
    Real x293 = x160 * x292;
    Real x294 = x106 * x284;
    Real x295 = x160 * x175;
    Real x296 = -x102 * x108 * x284;
    Real x297 = x141 * (-x176 + x177);
    Real x298 = x160 * x297;

    dV_t dW;

    dW.dA(0, 0) = (
        x120 * x162
        + x129 * x158
        + x157
        * (
            x116 * x126
            - x119 * x127
            - x128 * x133
            + x137 * x139
            + x140 * x152
            - x152 * x153
        )
        + x159 * x161
        + x164
    );
    dW.dA(0, 1) = (
        x108 * x165
        + x116 * x162
        + x137 * x161
        + x157
        * (
            x119 * x168
            - x120 * x126
            + x133 * x169
            - x139 * x159
            + x152 * x170
            - x152 * x171
            + x167
        )
    );
    dW.dA(0, 2) = (
        x102 * x165
        + x119 * x162
        + x133 * x160
        + x157
        * (
            -x116 * x168
            + x120 * x127
            - x137 * x175
            + x152 * x176
            - x152 * x177
            + x159 * x174
            + x173
        )
    );
    dW.dA(1, 0) = (
        x120 * x191
        + x157
        * (
            x116 * x194
            - x119 * x195
            + x139 * x196
            + x140 * x190
            - x153 * x190
            - x174 * x198
            + x192
        )
        + x158 * x179
        + x161 * x182
    );
    dW.dA(1, 1) = (
        x108 * x200
        + x116 * x191
        + x157
        * (
            x119 * x199
            - x120 * x194
            - x139 * x182
            + x170 * x190
            - x171 * x190
            + x175 * x198
        )
        + x161 * x196
        + x164
    );
    dW.dA(1, 2) = (
        x102 * x200
        + x119 * x191
        + x157
        * (
            -x116 * x199
            + x120 * x195
            + x174 * x182
            - x175 * x196
            + x176 * x190
            - x177 * x190
            + x201
        )
        + x161 * x198
    );
    dW.dA(2, 0) = (
        x120 * x215
        + x157
        * (
            x116 * x217
            - x119 * x218
            + x139 * x220
            + x140 * x214
            - x153 * x214
            + x172
            - x174 * x219
        )
        + x158 * x203
        + x161 * x207
    );
    dW.dA(2, 1) = (
        x108 * x221
        + x116 * x215
        + x157
        * (
            x119 * x223
            - x120 * x217
            - x139 * x207
            + x170 * x214
            - x171 * x214
            + x175 * x219
            + x222
        )
        + x161 * x220
    );
    dW.dA(2, 2) = (
        x102 * x221
        + x119 * x215
        + x157
        * (
            -x116 * x223
            + x120 * x218
            + x174 * x207
            - x175 * x220
            + x176 * x214
            - x177 * x214
        )
        + x161 * x219
        + x164
    );

    dW.dB(0, 0) = (
        x120 * x240
        + x157
        * (x139 * x247 + x140 * x239 - x153 * x239 - x174 * x248 + x243 - x246)
        + x158 * x225
        + x161 * x229
        + x224
    );
    dW.dB(0, 1) = (
        x116 * x240
        + x157
        * (
            x119 * x250
            - x120 * x242
            - x139 * x229
            + x170 * x239
            - x171 * x239
            + x175 * x248
            + x192
        )
        + x161 * x247
        + x249
    );
    dW.dB(0, 2) = (
        x119 * x240
        + x157
        * (
            -x116 * x250
            + x120 * x245
            + x172
            + x174 * x229
            - x175 * x247
            + x176 * x239
            - x177 * x239
        )
        + x161 * x248
        + x251
    );
    dW.dB(1, 0) = (
        x120 * x264
        + x157
        * (
            x116 * x266
            - x119 * x265
            + x139 * x267
            + x140 * x263
            - x153 * x263
            + x167
            - x174 * x268
        )
        + x161 * x256
        + x249
    );
    dW.dB(1, 1) = (
        x116 * x264
        + x157
        * (-x139 * x256 + x170 * x263 - x171 * x263 + x175 * x268 + x246 - x269)
        + x158 * x252
        + x161 * x267
        + x224
    );
    dW.dB(1, 2) = (
        x119 * x264
        + x157
        * (
            -x116 * x245
            + x120 * x265
            + x174 * x256
            - x175 * x267
            + x176 * x263
            - x177 * x263
            + x222
        )
        + x161 * x268
        + x270
    );
    dW.dB(2, 0) = (
        x120 * x278
        + x157
        * (
            x116 * x279
            - x119 * x266
            + x139 * x281
            + x140 * x277
            - x153 * x277
            + x173
            - x174 * x280
        )
        + x161 * x274
        + x251
    );
    dW.dB(2, 1) = (
        x116 * x278
        + x157
        * (
            x119 * x242
            - x120 * x279
            - x139 * x274
            + x170 * x277
            - x171 * x277
            + x175 * x280
            + x201
        )
        + x161 * x281
        + x270
    );
    dW.dB(2, 2) = (
        x119 * x278
        + x157
        * (x174 * x274 - x175 * x281 + x176 * x277 - x177 * x277 - x243 + x269)
        + x158 * x271
        + x161 * x280
        + x224
    );

    dW.dB0(0, 0) = x120 * x283 + x157 * (
        x140 * x282 - x153 * x282 + x285 + x286
    );
    dW.dB0(0, 1) = (
        x116 * x283 + x157 * (x170 * x282 - x171 * x282 + x289) + x287
    );
    dW.dB0(0, 2) = (
        x119 * x283 + x157 * (x176 * x282 - x177 * x282 + x291) - x290
    );
    dW.dB0(1, 0) = (
        x120 * x293 + x157 * (x140 * x292 - x153 * x292 + x289) - x287
    );
    dW.dB0(1, 1) = x116 * x293 + x157 * (
        x170 * x292 - x171 * x292 + x286 + x294
    );
    dW.dB0(1, 2) = (
        x119 * x293 + x157 * (x176 * x292 - x177 * x292 + x296) + x295
    );
    dW.dB0(2, 0) = (
        x120 * x298 + x157 * (x140 * x297 - x153 * x297 + x291) + x290
    );
    dW.dB0(2, 1) = (
        x116 * x298 + x157 * (x170 * x297 - x171 * x297 + x296) - x295
    );
    dW.dB0(2, 2) = x119 * x298 + x157 * (
        x176 * x297 - x177 * x297 + x285 + x294
    );
    // clang-format on

    return dW;
  }
};

template <typename Real>
struct lkball_globals {
  static constexpr Real overlap_gap_A2 = 0.5;
  static constexpr Real overlap_width_A2 = 2.6;
  static constexpr Real angle_overlap_A2 = 2.8 * overlap_width_A2;
  static constexpr Real ramp_width_A2 = 3.709;
};

template <typename Real, int MAX_WATER>
struct lk_fraction {
  typedef Eigen::Matrix<Real, 3, 1> Real3;
  typedef Eigen::Matrix<Real, MAX_WATER, 3> WatersMat;

  struct dV_t {
    WatersMat dWI;
    Real3 dJ;

    def astuple() { return make_tuple(dWI, dJ); }

    static def Zero()->dV_t { return {WatersMat::Zero(), Real3::Zero()}; }
  };

  static constexpr Real ramp_width_A2 = lkball_globals<Real>::ramp_width_A2;

  static def square(Real v)->Real { return v * v; }

  static def V(WatersMat WI, Real3 J, Real lj_radius_j)->Real {
    Real d2_low = std::max(0.0, square(1.4 + lj_radius_j) - ramp_width_A2);

    Real wted_d2_delta = 0;
    for (int w = 0; w < MAX_WATER; ++w) {
      Real d2_delta = (J - WI.row(w).transpose()).squaredNorm() - d2_low;
      if (!std::isnan(d2_delta)) {
        wted_d2_delta += std::exp(-d2_delta);
      }
    }

    wted_d2_delta = -std::log(wted_d2_delta);

    Real frac = 0;
    if (wted_d2_delta < 0) {
      frac = 1;
    } else if (wted_d2_delta < ramp_width_A2) {
      frac = square(1 - square(wted_d2_delta / ramp_width_A2));
    }

    return frac;
  }

  static def dV(WatersMat WI, Real3 J, Real lj_radius_j)->dV_t {
    Real d2_low = std::max(0.0, square(1.4 + lj_radius_j) - ramp_width_A2);

    Real wted_d2_delta = 0.0;
    Real3 d_wted_d2_delta_d_J = Real3::Zero();
    WatersMat d_wted_d2_delta_d_WI = WatersMat::Zero();

    for (int w = 0; w < MAX_WATER; ++w) {
      Real3 delta_Jw = J - WI.row(w).transpose();
      Real d2_delta = delta_Jw.squaredNorm() - d2_low;

      if (!std::isnan(d2_delta)) {
        Real exp_d2_delta = std::exp(-d2_delta);

        wted_d2_delta += exp_d2_delta;
        d_wted_d2_delta_d_J += 2 * exp_d2_delta * delta_Jw;
        d_wted_d2_delta_d_WI.row(w).transpose() = -2 * exp_d2_delta * delta_Jw;
      }
    }

    d_wted_d2_delta_d_J /= wted_d2_delta;
    d_wted_d2_delta_d_WI /= wted_d2_delta;

    wted_d2_delta = -std::log(wted_d2_delta);

    Real dfrac_dwted_d2 = 0;
    if (wted_d2_delta > 0 && wted_d2_delta < ramp_width_A2) {
      dfrac_dwted_d2 = -4.0 * wted_d2_delta
                       * (square(ramp_width_A2) - square(wted_d2_delta))
                       / square(square(ramp_width_A2));
    }

    return dV_t{d_wted_d2_delta_d_WI * dfrac_dwted_d2,
                d_wted_d2_delta_d_J * dfrac_dwted_d2};
  }
};

template <typename Real, int MAX_WATER>
struct lk_bridge_fraction {
  typedef Eigen::Matrix<Real, 3, 1> Real3;
  typedef Eigen::Matrix<Real, MAX_WATER, 3> WatersMat;

  struct dV_t {
    Real3 dI;
    Real3 dJ;
    WatersMat dWI;
    WatersMat dWJ;

    def astuple() { return make_tuple(dI, dJ, dWI, dWJ); }

    static def Zero()->dV_t {
      return dV_t{
          Real3::Zero(), Real3::Zero(), WatersMat::Zero(), WatersMat::Zero()};
    }
  };

  static constexpr Real ramp_width_A2 = lkball_globals<Real>::ramp_width_A2;
  static constexpr Real overlap_gap_A2 = lkball_globals<Real>::overlap_gap_A2;
  static constexpr Real overlap_width_A2 =
      lkball_globals<Real>::overlap_width_A2;
  static constexpr Real angle_overlap_A2 =
      lkball_globals<Real>::angle_overlap_A2;

  static def square(Real v)->Real { return v * v; }

  static def V(
      Real3 I, Real3 J, WatersMat WI, WatersMat WJ, Real lkb_water_dist)
      ->Real {
    // water overlap
    Real wted_d2_delta = 0.0;
    for (int wi = 0; wi < MAX_WATER; wi++) {
      for (int wj = 0; wj < MAX_WATER; wj++) {
        Real d2_delta =
            (WI.row(wi) - WJ.row(wj)).squaredNorm() - overlap_gap_A2;
        if (!std::isnan(d2_delta)) {
          wted_d2_delta += std::exp(-d2_delta);
        }
      }
    }
    wted_d2_delta = -std::log(wted_d2_delta);

    Real overlapfrac;
    if (wted_d2_delta > overlap_width_A2) {
      overlapfrac = 0;
    } else {
      // square-square -> 1 as x -> 0
      overlapfrac = square(1 - square(wted_d2_delta / overlap_width_A2));
    }

    // base angle
    Real overlap_target_len2 = 8.0 / 3.0 * square(lkb_water_dist);
    Real overlap_len2 = (I - J).squaredNorm();
    Real base_delta = std::abs(overlap_len2 - overlap_target_len2);

    Real anglefrac;
    if (base_delta > angle_overlap_A2) {
      anglefrac = 0;
    } else {
      // square-square -> 1 as x -> 0
      anglefrac = square(1 - square(base_delta / angle_overlap_A2));
    }

    return overlapfrac * anglefrac;
  }

  static def dV(
      Real3 I, Real3 J, WatersMat WI, WatersMat WJ, Real lkb_water_dist)
      ->dV_t {
    Real wted_d2_delta = 0.0;

    WatersMat d_wted_d2_delta_d_WI = WatersMat::Zero();
    WatersMat d_wted_d2_delta_d_WJ = WatersMat::Zero();

    for (int wi = 0; wi < MAX_WATER; wi++) {
      for (int wj = 0; wj < MAX_WATER; wj++) {
        Real3 delta_ij = WI.row(wi) - WJ.row(wj);
        Real d2_delta = delta_ij.squaredNorm() - overlap_gap_A2;

        if (!std::isnan(d2_delta)) {
          Real exp_d2_delta = std::exp(-d2_delta);

          d_wted_d2_delta_d_WI.row(wi).transpose() +=
              2 * exp_d2_delta * delta_ij;
          d_wted_d2_delta_d_WJ.row(wj).transpose() -=
              2 * exp_d2_delta * delta_ij;

          wted_d2_delta += std::exp(-d2_delta);
        }
      }
    }

    if (wted_d2_delta != 0) {
      d_wted_d2_delta_d_WI /= wted_d2_delta;
      d_wted_d2_delta_d_WJ /= wted_d2_delta;
    }

    wted_d2_delta = -std::log(wted_d2_delta);

    Real overlapfrac;
    Real d_overlapfrac_d_wted_d2;

    if (wted_d2_delta > overlap_width_A2) {
      overlapfrac = 0.0;
      d_overlapfrac_d_wted_d2 = 0.0;
    } else if (wted_d2_delta > 0) {
      overlapfrac = square(1 - square(wted_d2_delta / overlap_width_A2));
      d_overlapfrac_d_wted_d2 =
          -4.0 * wted_d2_delta
          * (square(overlap_width_A2) - square(wted_d2_delta))
          / square(square(overlap_width_A2));
    } else {
      overlapfrac = 1.0;
      d_overlapfrac_d_wted_d2 = 0.0;
    }

    // base angle
    Real overlap_target_len2 = 8.0 / 3.0 * square(lkb_water_dist);
    Real3 delta_ij = I - J;
    Real overlap_len2 = delta_ij.squaredNorm();
    Real base_delta = overlap_len2 - overlap_target_len2;
    Real3 d_wted_d2_delta_d_I = 2.0 * delta_ij;
    Real3 d_wted_d2_delta_d_J = -2.0 * delta_ij;

    Real anglefrac;
    Real d_anglefrac_d_base_delta;
    if (std::abs(base_delta) > angle_overlap_A2) {
      anglefrac = 0.0;
      d_anglefrac_d_base_delta = 0.0;
    } else if (std::abs(base_delta) > 0.0) {
      anglefrac = square(1 - square(base_delta / angle_overlap_A2));
      d_anglefrac_d_base_delta =
          -4.0 * base_delta * (square(angle_overlap_A2) - square(base_delta))
          / square(square(angle_overlap_A2));
    } else {
      anglefrac = 1.0;
      d_anglefrac_d_base_delta = 0.0;
    }

    // final scaling
    return dV_t{
        overlapfrac * d_anglefrac_d_base_delta * d_wted_d2_delta_d_I,
        overlapfrac * d_anglefrac_d_base_delta * d_wted_d2_delta_d_J,
        anglefrac * d_overlapfrac_d_wted_d2 * d_wted_d2_delta_d_WI,
        anglefrac * d_overlapfrac_d_wted_d2 * d_wted_d2_delta_d_WJ,
    };
  }
};

template <typename Real, int MAX_WATER>
struct lk_ball_score {
  typedef Eigen::Matrix<Real, 3, 1> Real3;
  typedef Eigen::Matrix<Real, MAX_WATER, 3> WatersMat;

  struct V_t {
    Real lkball_iso;
    Real lkball;
    Real lkbridge;
    Real lkbridge_uncpl;

    auto astuple() {
      return make_tuple(lkball_iso, lkball, lkbridge, lkbridge_uncpl);
    }
  };

  struct dV_dReal3 {
    Real3 d_lkball_iso;
    Real3 d_lkball;
    Real3 d_lkbridge;
    Real3 d_lkbridge_uncpl;

    auto astuple() {
      return make_tuple(d_lkball_iso, d_lkball, d_lkbridge, d_lkbridge_uncpl);
    }

    static def Zero()->dV_dReal3 {
      return {Real3::Zero(), Real3::Zero(), Real3::Zero(), Real3::Zero()};
    }
  };

  struct dV_dWatersMat {
    WatersMat d_lkball_iso;
    WatersMat d_lkball;
    WatersMat d_lkbridge;
    WatersMat d_lkbridge_uncpl;

    auto astuple() {
      return make_tuple(d_lkball_iso, d_lkball, d_lkbridge, d_lkbridge_uncpl);
    }

    static def Zero()->dV_dWatersMat {
      return {WatersMat::Zero(),
              WatersMat::Zero(),
              WatersMat::Zero(),
              WatersMat::Zero()};
    }
  };

  struct dV_t {
    dV_dReal3 dI;
    dV_dReal3 dJ;
    dV_dWatersMat dWI;
    dV_dWatersMat dWJ;

    auto astuple() {
      return make_tuple(
          dI.astuple(), dJ.astuple(), dWI.astuple(), dWJ.astuple());
    }

    static def Zero()->dV_t {
      return {dV_dReal3::Zero(),
              dV_dReal3::Zero(),
              dV_dWatersMat::Zero(),
              dV_dWatersMat::Zero()};
    }
  };

  static def V(
      Real3 I,
      Real3 J,
      WatersMat WI,
      WatersMat WJ,
      Real bonded_path_length,
      Real lkb_water_dist,
      LKTypeParams<Real> i,
      LKTypeParams<Real> j,
      LJGlobalParams<Real> global)
      ->V_t {
    Real sigma = lj_sigma<Real>(i, j, global);

    Real dist = distance<Real>::V(I, J);

    Real lk_iso_IJ = lk_isotropic_pair_V<Real>(
        dist,
        bonded_path_length,
        sigma,
        i.lj_radius,
        i.lk_dgfree,
        i.lk_lambda,
        j.lk_volume);

    Real frac_IJ_desolv = lk_fraction<Real, MAX_WATER>::V(WI, J, j.lj_radius);

    Real frac_IJ_water_overlap;
    if (j.is_donor || j.is_acceptor) {
      frac_IJ_water_overlap =
          lk_bridge_fraction<Real, MAX_WATER>::V(I, J, WI, WJ, lkb_water_dist);
    } else {
      frac_IJ_water_overlap = 0.0;
    }

    return V_t{
        lk_iso_IJ,
        lk_iso_IJ * frac_IJ_desolv,
        lk_iso_IJ * frac_IJ_water_overlap,
        frac_IJ_water_overlap / 2,
    };
  }

  static def dV(
      Real3 I,
      Real3 J,
      WatersMat WI,
      WatersMat WJ,
      Real bonded_path_length,
      Real lkb_water_dist,
      LKTypeParams<Real> i,
      LKTypeParams<Real> j,
      LJGlobalParams<Real> global)
      ->dV_t {
    Real sigma = lj_sigma<Real>(i, j, global);

    auto _dist = distance<Real>::V_dV(I, J);
    Real dist = _dist.V;
    Real3 d_dist_dI = _dist.dV_dA;
    Real3 d_dist_dJ = _dist.dV_dB;

    auto _lk_iso_IJ = lk_isotropic_pair_V_dV<Real>(
        dist,
        bonded_path_length,
        sigma,
        i.lj_radius,
        i.lk_dgfree,
        i.lk_lambda,
        j.lk_volume);
    Real lk_iso_IJ = get<0>(_lk_iso_IJ);
    Real d_lk_iso_IJ_d_dist = get<1>(_lk_iso_IJ);

    Real frac_IJ_desolv = lk_fraction<Real, MAX_WATER>::V(WI, J, j.lj_radius);
    auto d_frac_IJ_desolv =
        lk_fraction<Real, MAX_WATER>::dV(WI, J, j.lj_radius);

    Real frac_IJ_water_overlap;
    typename lk_bridge_fraction<Real, MAX_WATER>::dV_t d_frac_IJ_water_overlap;
    if (j.is_donor || j.is_acceptor) {
      frac_IJ_water_overlap =
          lk_bridge_fraction<Real, MAX_WATER>::V(I, J, WI, WJ, lkb_water_dist);
      d_frac_IJ_water_overlap =
          lk_bridge_fraction<Real, MAX_WATER>::dV(I, J, WI, WJ, lkb_water_dist);
    } else {
      frac_IJ_water_overlap = 0.0;
      d_frac_IJ_water_overlap = decltype(d_frac_IJ_water_overlap)::Zero();
    }

    Real3 d_lk_iso_IJ_dI = d_lk_iso_IJ_d_dist * d_dist_dI;
    Real3 d_lk_iso_IJ_dJ = d_lk_iso_IJ_d_dist * d_dist_dJ;

    return dV_t{
        // dI
        dV_dReal3{
            d_lk_iso_IJ_dI,
            // d(lk_iso_IJ * frac_IJ_desolv)
            // d(frac_IJ_desolv)/dI == 0
            d_lk_iso_IJ_dI * frac_IJ_desolv,
            // d(lk_iso_IJ * frac_IJ_water_overlap)
            d_lk_iso_IJ_dI * frac_IJ_water_overlap
                + lk_iso_IJ * d_frac_IJ_water_overlap.dI,
            // d(frac_IJ_water_overlap / 2)
            d_frac_IJ_water_overlap.dI / 2,
        },
        // dJ
        dV_dReal3{
            // d(lk_iso_IJ)
            d_lk_iso_IJ_dJ,
            // d(lk_iso_IJ * frac_IJ_desolv)
            d_lk_iso_IJ_dJ * frac_IJ_desolv + lk_iso_IJ * d_frac_IJ_desolv.dJ,
            // d(lk_iso_IJ * frac_IJ_water_overlap)
            d_lk_iso_IJ_dJ * frac_IJ_water_overlap
                + lk_iso_IJ * d_frac_IJ_water_overlap.dJ,
            // d(frac_IJ_water_overlap / 2)
            d_frac_IJ_water_overlap.dJ / 2,
        },
        // dWI
        dV_dWatersMat{// d(lk_iso_IJ)
                      WatersMat::Zero(),

                      // d(lk_iso_IJ * frac_IJ_desolv)
                      // d(lk_iso_IJ)/dWI == 0
                      lk_iso_IJ * d_frac_IJ_desolv.dWI,

                      // d(lk_iso_IJ * frac_IJ_water_overlap)
                      // d(lk_iso_IJ)/dWI == 0
                      lk_iso_IJ * d_frac_IJ_water_overlap.dWI,

                      // d(frac_IJ_water_overlap / 2)
                      d_frac_IJ_water_overlap.dWI / 2},
        // dWJ
        dV_dWatersMat{// d(lk_iso_IJ)
                      WatersMat::Zero(),

                      // d(lk_iso_IJ * frac_IJ_desolv)
                      WatersMat::Zero(),

                      // d(lk_iso_IJ * frac_IJ_water_overlap)
                      // d(lk_iso_IJ)/dWJ == 0
                      lk_iso_IJ * d_frac_IJ_water_overlap.dWJ,

                      // d(frac_IJ_water_overlap / 2)
                      d_frac_IJ_water_overlap.dWJ / 2}};
  }
};

#undef def

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
