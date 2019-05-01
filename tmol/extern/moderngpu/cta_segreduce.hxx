// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once

#include "cta_scan.hxx"
#include "cta_segscan.hxx"

BEGIN_MGPU_NAMESPACE


////////////////////////////////////////////////////////////////////////////////
// cta_segreduce_t is common intra-warp segmented reduction code for 
// these kernels. Should clean up and move to cta_segreduce.hxx.

template<int nt, int vt, typename type_t>
struct cta_segreduce_t {
  typedef cta_segscan_t<nt, type_t> segscan_t;
  
  union storage_t {
    typename segscan_t::storage_t segscan;
    type_t values[nt * vt + 1];
  };

  // Values must be stored in storage.values on entry.
  template<typename op_t, typename output_it>
  MGPU_DEVICE void segreduce(merge_range_t merge_range, 
    lbs_placement_t placement, array_t<bool, vt + 1> p, int tid, 
    int cta, type_t init, op_t op, output_it output, 
    type_t* carry_out_values, int* carry_out_codes, storage_t& storage) {

    int cur_item = placement.a_index;
    int begin_segment = placement.b_index;
    int cur_segment = begin_segment;
    bool carry_in = false;

    const type_t* a_shared = storage.values - merge_range.a_begin;
    type_t x[vt];
    int segments[vt + 1];
    iterate<vt>([&](int i) {
      if(p[i]) {
        // This is a data node, so accumulate and advance the data ID.
        x[i] = a_shared[cur_item++];
        if(carry_in) x[i] = op(x[i - 1], x[i]);
        carry_in = true;
      } else {
        // This is a segment node, so advance the segment ID.
        x[i] = init;
        ++cur_segment;
        carry_in = false;
      }
      segments[i] = cur_segment;
    });
    // Always flush at the end of the last thread.
    bool overwrite = (nt - 1 == tid) && (!p[vt - 1] && p[vt]);
    if(nt - 1 == tid) p[vt] = false;
    if(!p[vt]) ++cur_segment;
    segments[vt] = cur_segment;
    overwrite = __syncthreads_or(overwrite);

    // Get the segment ID for the next item. This lets us find an end flag
    // for the last value in this thread.
    bool has_head_flag = begin_segment < segments[vt - 1];
    bool has_carry_out = p[vt - 1];

    // Compute the carry-in for each thread.
    segscan_result_t<type_t> result = segscan_t().segscan(tid, has_head_flag,
      has_carry_out, x[vt - 1], storage.segscan, init, op);

    // Add the carry-in back into each value and recompute the reductions.
    type_t* x_shared = storage.values - placement.range.b_begin;
    carry_in = result.has_carry_in && p[0];
    iterate<vt>([&](int i) {
      if(segments[i] < segments[i + 1]) {
        // We've hit the end of this segment. Store the reduction to shared
        // memory.
        if(carry_in) x[i] = op(result.scan, x[i]);
        x_shared[segments[i]] = x[i];
        carry_in = false;
      }
    });
    __syncthreads();

    // Store the reductions for segments which begin in this tile. 
    for(int i = merge_range.b_begin + tid; i < merge_range.b_end; i += nt)
      output[i] = x_shared[i];

    // Store the partial reduction for the segment which begins in the 
    // preceding tile, if there is one.
    if(!tid) {
      if(segments[0] == merge_range.b_begin) segments[0] = -1;
      int code = (segments[0]<< 1) | (int)overwrite;
      carry_out_values[cta] = (segments[0] != -1) ?
        x_shared[segments[0]] : 
        init;
      carry_out_codes[cta] = code;
    }
  }
};


END_MGPU_NAMESPACE
