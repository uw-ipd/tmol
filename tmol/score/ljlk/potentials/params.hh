#define _STR(v) #v

template <typename Real>
struct LJTypeParams {
  Real lj_radius;
  Real lj_wdepth;
  bool is_donor;
  bool is_hydroxyl;
  bool is_polarh;
  bool is_acceptor;
};

// clang-format off
//
#define LJTypeParams_args(PREFIX)    \
  Real PREFIX##lj_radius,   \
  Real PREFIX##lj_wdepth,   \
  bool PREFIX##is_donor,    \
  bool PREFIX##is_hydroxyl, \
  bool PREFIX##is_polarh,   \
  bool PREFIX##is_acceptor

#define LJTypeParams_targs(NDIM)    \
  TView<Real, NDIM> lj_radius,   \
  TView<Real, NDIM> lj_wdepth,   \
  TView<bool, NDIM> is_donor,    \
  TView<bool, NDIM> is_hydroxyl, \
  TView<bool, NDIM> is_polarh,   \
  TView<bool, NDIM> is_acceptor

#define LJTypeParams_struct(PREFIX, ...)   \
{                  \
  PREFIX##lj_radius __VA_ARGS__,   \
  PREFIX##lj_wdepth __VA_ARGS__,   \
  PREFIX##is_donor __VA_ARGS__,    \
  PREFIX##is_hydroxyl __VA_ARGS__, \
  PREFIX##is_polarh __VA_ARGS__,   \
  PREFIX##is_acceptor __VA_ARGS__  \
}

#define LJTypeParams_pyargs(PREFIX)      \
  py::arg( _STR(PREFIX##lj_radius) ),   \
  py::arg( _STR(PREFIX##lj_wdepth) ),   \
  py::arg( _STR(PREFIX##is_donor) ),    \
  py::arg( _STR(PREFIX##is_hydroxyl) ), \
  py::arg( _STR(PREFIX##is_polarh) ),   \
  py::arg( _STR(PREFIX##is_acceptor) )
// clang-format on

template <typename Real>
struct LKTypeParams {
  Real lj_radius;
  Real lk_dgfree;
  Real lk_lambda;
  Real lk_volume;
  bool is_donor;
  bool is_hydroxyl;
  bool is_polarh;
  bool is_acceptor;
};

// clang-format off
#define LKTypeParams_args(PREFIX)    \
  Real PREFIX##lj_radius,   \
  Real PREFIX##lk_dgfree,   \
  Real PREFIX##lk_lambda,   \
  Real PREFIX##lk_volume,   \
  bool PREFIX##is_donor,    \
  bool PREFIX##is_hydroxyl, \
  bool PREFIX##is_polarh,   \
  bool PREFIX##is_acceptor

#define LKTypeParams_targs(NDIM)    \
  TView<Real, NDIM> lj_radius,   \
  TView<Real, NDIM> lk_dgfree,   \
  TView<Real, NDIM> lk_lambda,   \
  TView<Real, NDIM> lk_volume,   \
  TView<bool, NDIM> is_donor,    \
  TView<bool, NDIM> is_hydroxyl, \
  TView<bool, NDIM> is_polarh,   \
  TView<bool, NDIM> is_acceptor

#define LKTypeParams_struct(PREFIX, ...)   \
{                  \
  PREFIX##lj_radius __VA_ARGS__,   \
  PREFIX##lk_dgfree __VA_ARGS__,   \
  PREFIX##lk_lambda __VA_ARGS__,   \
  PREFIX##lk_volume __VA_ARGS__,   \
  PREFIX##is_donor __VA_ARGS__,    \
  PREFIX##is_hydroxyl __VA_ARGS__, \
  PREFIX##is_polarh __VA_ARGS__,   \
  PREFIX##is_acceptor __VA_ARGS__  \
}

#define LKTypeParams_pyargs(PREFIX)     \
  py::arg( _STR(PREFIX##lj_radius) ),  \
  py::arg( _STR(PREFIX##lk_dgfree) ),  \
  py::arg( _STR(PREFIX##lk_lambda) ),  \
  py::arg( _STR(PREFIX##lk_volume) ),  \
  py::arg( _STR(PREFIX##is_donor) ),   \
  py::arg( _STR(PREFIX##is_hydroxyl) ),\
  py::arg( _STR(PREFIX##is_polarh) ),  \
  py::arg( _STR(PREFIX##is_acceptor) )
// clang-format on

template <typename Real>
struct LJGlobalParams {
  Real lj_hbond_dis;
  Real lj_hbond_OH_donor_dis;
  Real lj_hbond_hdis;
};

// clang-format off
#define LJGlobalParams_args() \
  Real lj_hbond_dis,          \
  Real lj_hbond_OH_donor_dis, \
  Real lj_hbond_hdis          \

#define LJGlobalParams_pyargs()           \
  py::arg( _STR(lj_hbond_dis) ),          \
  py::arg( _STR(lj_hbond_OH_donor_dis) ), \
  py::arg( _STR(lj_hbond_hdis) )


#define LJGlobalParams_struct() \
{                               \
  lj_hbond_dis,                 \
  lj_hbond_OH_donor_dis,        \
  lj_hbond_hdis                 \
}
// clang-format on
