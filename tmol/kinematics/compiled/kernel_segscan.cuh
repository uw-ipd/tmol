#include <moderngpu/cta_load_balance.hxx>
#include <moderngpu/cta_reduce.hxx>
#include <moderngpu/cta_scan.hxx>
#include <moderngpu/cta_segreduce.hxx>
#include <moderngpu/cta_segscan.hxx>
#include <moderngpu/memory.hxx>
#include <moderngpu/search.hxx>
#include <moderngpu/transform.hxx>

namespace tmol {
namespace kinematics {

using namespace mgpu;

// The per-thread code for the segmented scan
template <int nt, int vt, typename type_t>
struct cta_lbs_segscan_t {
  typedef cta_segscan_t<nt, type_t> segscan_t;
  typedef cta_reduce_t<nt, int> reduce_t;  // reduction is over indices

  union storage_t {
    typename segscan_t::storage_t segscan;
    typename reduce_t::storage_t reduce;
    type_t values[nt * vt + 1];
  };

  // Values must be stored in storage.values on entry.
  template <typename op_t, typename output_it>
  MGPU_DEVICE void segscan(
      merge_range_t merge_range,
      lbs_placement_t placement,
      array_t<bool, vt + 1> p,
      int tid,
      int cta,
      type_t init,
      op_t op,
      output_it output,
      type_t* carry_out_values,
      int* carry_out_codes,
      storage_t& storage,
      scan_type_t type = scan_type_inc) {
    // scan through workunits for this thread
    int cur_item = placement.a_index;
    bool carry_in = false;
    const type_t* a_shared = storage.values - merge_range.a_begin;
    type_t x[vt];
    bool has_head_flag = false;
    int stopindex = nt * vt + 1;
    iterate<vt>([&](int i) {
      if (p[i]) {
        // This is a data node, so accumulate and advance the data ID.
        x[i] = a_shared[cur_item++];
        if (carry_in) x[i] = op(x[i - 1], x[i]);
        carry_in = true;
      } else {
        // This is a segment node, so advance the segment ID.
        x[i] = init;
        carry_in = false;
        if (stopindex == nt * vt + 1) {
          stopindex = vt * tid + i;  // thread index
        }
        has_head_flag = true;
      }
    });

    // is there a segment transition in this range?
    bool segment_has_head_flag = __syncthreads_or(has_head_flag);

    // find the first segment start in this cta
    stopindex = reduce_t().reduce(
        tid, stopindex, storage.reduce, nt, minimum_t<int>(), false);

    // compute the carry-in for this thread
    bool has_carry_out = p[vt - 1];
    segscan_result_t<type_t> result = segscan_t().segscan(
        tid,
        has_head_flag,
        has_carry_out,
        x[vt - 1],
        storage.segscan,
        init,
        op);

    // add carry-in back to each value
    cur_item = placement.a_index;
    carry_in = tid > 0;
    iterate<vt>([&](int i) {
      if (p[i]) {
        if (type == scan_type_inc) {
          output[cur_item++] = carry_in ? op(result.scan, x[i]) : x[i];
        } else {
          output[cur_item++] = carry_in ? result.scan : init;
        }
      } else {
        carry_in = false;
      }
    });

    // store the carryout of this scan (thread nt-1)
    if (tid == nt - 1) {
      // p[vt-1] checks if the current node (the last of the cta)
      //   is a segment node.  If so, no carryout.
      if (p[vt - 1]) {
        carry_out_values[cta] = (type == scan_type_inc)
                                    ? output[cur_item - 1]
                                    : op(output[cur_item - 1], x[vt - 1]);
      } else {
        carry_out_values[cta] = init;
      }
    }

    // store when the first input scan stops
    if (tid == 0) {
      carry_out_codes[cta] = (stopindex << 1) + segment_has_head_flag;
    }
  }
};

// The "spine" scanning code
//   given the outputs of each segscan cta and the "passthrough"
//     flag, compute the input to each group
//   recursive call in case #nodes > 512*512 (or whatever)
//     with both upward- & downward-pass ctas
template <int nt, typename type_t, typename op_t>
void spine_segreduce(
    type_t* values,
    const int* codes,
    int count,
    op_t op,
    type_t init,
    context_t& context) {
  int num_ctas = div_up(count, nt);

  mem_t<type_t> carry_out(num_ctas, context);
  mem_t<int> codes_out(num_ctas, context);
  type_t* carry_out_values = carry_out.data();
  int* carry_out_codes = codes_out.data();

  // upward pass
  auto k_spine_up = [=] MGPU_DEVICE(int tid, int cta) {
    typedef cta_segscan_t<nt, type_t> segscan_t;
    typedef cta_reduce_t<nt, int> reduce_t;  // reduction is over indices

    __shared__ union {
      typename segscan_t::storage_t segscan;
      typename reduce_t::storage_t reduce;
    } shared;

    range_t tile = get_tile(cta, nt, count);
    int gid = tile.begin + tid;

    type_t value = (gid < count) ? values[gid] : init;
    bool has_head_flag = (gid < count) ? (codes[gid] & 1) : false;
    bool has_carry_out = (gid < count);

    segscan_result_t<type_t> result = segscan_t().segscan(
        tid, has_head_flag, has_carry_out, value, shared.segscan, init, op);
    if (gid < count) {
      values[gid] = result.scan;
    }

    // find the first segment start in this cta
    int stopindex = has_head_flag ? tid : nt + 1;
    stopindex = reduce_t().reduce(
        tid,
        stopindex,
        shared.reduce,
        min(tile.count(), (int)nt),
        minimum_t<int>(),
        false);

    bool segment_has_head_flag = __syncthreads_or(has_head_flag);
    if (tid == nt - 1) {
      carry_out_values[cta] = result.reduction;
    }
    if (tid == 0) {
      carry_out_codes[cta] = (stopindex << 1) + segment_has_head_flag;
    }
  };
  cta_launch<nt>(k_spine_up, num_ctas, context);

  // recursive call if there are > 1 ctas
  if (num_ctas > 1) {
    spine_segreduce<nt>(
        carry_out_values, carry_out_codes, num_ctas, op, init, context);

    // downward pass (not cta 0)
    auto k_spine_down = [=] MGPU_DEVICE(int tid, int cta) {
      range_t tile = get_tile(cta + 1, nt, count);
      int gid = tile.begin + tid;

      int seg0 = carry_out_codes[cta + 1] >> 1;
      if (tid <= seg0 && gid < count) {
        values[gid] = op(carry_out_values[cta + 1], values[gid]);
      }
    };
    cta_launch<nt>(k_spine_down, num_ctas - 1, context);
  }
}

// segscan kernel
template <
    typename launch_arg_t = empty_t,
    scan_type_t scan_type = scan_type_inc,
    typename func_t,
    typename segments_it,
    typename output_it,
    typename op_t,
    typename type_t>
void kernel_segscan(
    func_t f,
    int count,
    segments_it segments,
    int num_segments,
    output_it output,
    op_t op,
    type_t init,
    context_t& context) {
  typedef typename conditional_typedef_t<
      launch_arg_t,
      launch_box_t<
          arch_20_cta<128, 11, 8>,
          arch_35_cta<128, 7, 5>,
          arch_52_cta<128, 11, 8> > >::type_t launch_t;

  cta_dim_t cta_dim = launch_t::cta_dim(context);
  int num_ctas = cta_dim.num_ctas(count + num_segments);

  // storage between temp arrays
  mem_t<type_t> carry_out(num_ctas, context);
  mem_t<int> codes(num_ctas, context);
  type_t* carry_out_data = carry_out.data();
  int* codes_data = codes.data();

  mem_t<int> mp = load_balance_partitions(
      count, segments, num_segments, cta_dim.nv(), context);
  const int* mp_data = mp.data();

  // "upward" scan:
  //   - within each CTA, run the forward segscan
  //   - compute the value to be passed to the next CTA (carry_out_data)
  //   - compute whether or not this segment
  auto k_scan = [=] MGPU_DEVICE(int tid, int cta) {
    typedef typename launch_t::sm_ptx params_t;
    enum { nt = params_t::nt, vt = params_t::vt, vt0 = params_t::vt0 };
    typedef cta_load_balance_t<nt, vt> load_balance_t;
    typedef cta_lbs_segscan_t<nt, vt, type_t> lbs_segscan_t;

    __shared__ union {
      typename load_balance_t::storage_t lbs;
      typename lbs_segscan_t::storage_t mysegscan;
      type_t values[nt * vt + 1];
    } shared;

    // Compute the load-balancing search and materialize (index, seg, rank)
    // arrays.
    auto lbs = load_balance_t().load_balance(
        count, segments, num_segments, tid, cta, mp_data, shared.lbs);

    // Call the user-supplied functor f.
    array_t<type_t, vt> strided_values;
    strided_iterate<nt, vt, vt0>(
        [&](int i, int j) {
          int index = lbs.merge_range.a_begin + j;
          int seg = lbs.segments[i];
          int rank = lbs.ranks[i];
          strided_values[i] = f(index, seg, rank);
        },
        tid,
        lbs.merge_range.a_count());

    // Store the values back to shared memory for segmented scan
    reg_to_shared_strided<nt, vt>(strided_values, tid, shared.mysegscan.values);

    // Split the flags.
    array_t<bool, vt + 1> merge_bits;
    iterate<vt + 1>(
        [&](int i) { merge_bits[i] = 0 != ((1 << i) & lbs.merge_flags); });

    cta_lbs_segscan_t<nt, vt, type_t>().segscan(
        lbs.merge_range,
        lbs.placement,
        merge_bits,
        tid,
        cta,
        init,
        op,
        output,
        carry_out_data,
        codes_data,
        shared.mysegscan,
        scan_type);
  };
  cta_launch<launch_t>(k_scan, num_ctas, context);

  // if there was only 1 CTA, we're done
  if (num_ctas == 1) {
    return;
  }

  // perform segscans across "carry_out"
  mem_t<type_t> carry_out_scan(num_ctas, context);
  spine_segreduce<launch_t::nt>(
      carry_out_data, codes_data, num_ctas, op, init, context);

  // final downward sweep
  auto k_finalsweep = [=] MGPU_DEVICE(int tid, int cta) {
    typedef typename launch_t::sm_ptx params_t;
    enum { nt = params_t::nt, vt = params_t::vt, vt0 = params_t::vt0 };
    typedef cta_load_balance_t<nt, vt> load_balance_t;

    __shared__ union { typename load_balance_t::storage_t lbs; } shared;

    // Need to rematerialize load balance parameters
    auto lbs = load_balance_t().load_balance(
        count, segments, num_segments, tid, cta + 1, mp_data, shared.lbs);

    // scan through workunits for this thread
    int seg0 = codes_data[cta + 1] >> 1;
    int cur_item = lbs.placement.a_index;
    iterate<vt>([&](int i) {
      if (((1 << i) & lbs.merge_flags) != 0) {
        // this is a data node
        if (vt * tid + i < seg0) {
          output[cur_item] = op(carry_out_data[cta + 1], output[cur_item]);
          cur_item++;
        }
      }
    });
  };
  cta_launch<launch_t>(k_finalsweep, num_ctas - 1, context);
}

}  // namespace kinematics
}  // namespace tmol