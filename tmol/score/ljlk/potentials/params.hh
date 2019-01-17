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

#define LJTypeParams_struct(PREFIX)   \
{                  \
  PREFIX##lj_radius,   \
  PREFIX##lj_wdepth,   \
  PREFIX##is_donor,    \
  PREFIX##is_hydroxyl, \
  PREFIX##is_polarh,   \
  PREFIX##is_acceptor  \
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

#define LKTypeParams_struct(PREFIX)   \
{                  \
  PREFIX##lj_radius,   \
  PREFIX##lk_dgfree,   \
  PREFIX##lk_lambda,   \
  PREFIX##lk_volume,   \
  PREFIX##is_donor,    \
  PREFIX##is_hydroxyl, \
  PREFIX##is_polarh,   \
  PREFIX##is_acceptor  \
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
