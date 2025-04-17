#pragma once

// For mgpu namespace macros
#include "meta.hxx"

BEGIN_MGPU_NAMESPACE

// Types for scan operations that are CPU-compatible.

enum scan_type_t {
  scan_type_exc,
  scan_type_inc
};

END_MGPU_NAMESPACE
