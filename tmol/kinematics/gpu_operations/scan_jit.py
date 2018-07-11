import math
import numba
import numba.cuda as cuda


class GenerationalSegmentedScan:
    threads_per_block: int
    _segscan_by_generation: object

    def segscan_by_generation(
            self,
            src_vals,  # [n] + [val_shape]
            scan_to_src_ordering,  # [n]
            is_path_root,  #[n]
            non_path_inputs,  #[n, max_num_inputs]
            generation_ranges,  #[g, 2]
    ):
        self._segscan_by_generation[1, self.threads_per_block](
            src_vals,
            scan_to_src_ordering,
            is_path_root,
            non_path_inputs,
            generation_ranges,
        )

    def __init__(self, threads_per_block=256):
        if not math.log2(threads_per_block).is_integer():
            raise ValueError(
                f"thread_per_block must be power of 2: {threads_per_block}"
            )

        add = self.add
        zero = self.zero
        load = self.load
        save = self.save

        shared_shape = (threads_per_block, ) + self.val_shape

        n_scan_iter = int(math.log2(threads_per_block))

        @cuda.jit
        def _segscan_by_generation(
                src_vals,  # [n] + [val_shape]
                scan_to_src_ordering,  # [n]
                is_path_root,  #[n]
                non_path_inputs,  #[n, max_num_inputs]
                generation_ranges,  #[g, 2]
        ):
            shared_vals = cuda.shared.array(shared_shape, numba.float64)
            shared_is_root = cuda.shared.array((threads_per_block),
                                               numba.int32)

            pos = cuda.grid(1)

            for gen in range(generation_ranges.shape[0]):
                start = generation_ranges[gen, 0]
                end = generation_ranges[gen, 1]
                blocks_for_gen = (end - start - 1) // threads_per_block + 1

                ### Iterate block across generation
                carry_val = zero()
                carry_is_root = False

                for ii in range(blocks_for_gen):
                    ii_ind = ii * threads_per_block + start + pos
                    ii_src = -1

                    ### Load shared memory value block in scan order
                    if ii_ind < end:
                        ii_src = scan_to_src_ordering[ii_ind]

                        ### Read node values from global into shared
                        my_val = load(src_vals, ii_src)
                        shared_is_root[pos] = is_path_root[ii_ind]

                        ### Sum incoming scan value from parent into node
                        # parent only set if node is root of scan
                        for jj in range(non_path_inputs.shape[1]):
                            input_ind = non_path_inputs[ii_ind, jj]
                            if input_ind != -1:
                                my_val = add(
                                    load(
                                        src_vals,
                                        scan_to_src_ordering[input_ind]
                                    ), my_val
                                )

                        ### Sum carry value from previous block if node 0 is non-root.
                        my_root = shared_is_root[pos]
                        if pos == 0 and not my_root:
                            my_val = add(carry_val, my_val)
                            my_root |= carry_is_root
                            shared_is_root[0] = my_root

                        save(shared_vals, pos, my_val)

                    ### Sync on prepared shared memory block
                    cuda.syncthreads()

                    ### Perform parallel segmented scan on block
                    offset = 1
                    for _ in range(n_scan_iter):

                        if pos >= offset and ii_ind < end:
                            prev_val = load(shared_vals, pos - offset)
                            prev_root = shared_is_root[pos - offset]
                        cuda.syncthreads()

                        if pos >= offset and ii_ind < end:
                            if not my_root:
                                my_val = add(prev_val, my_val)
                                my_root |= prev_root
                                save(shared_vals, pos, my_val)
                                shared_is_root[pos] = my_root
                        offset *= 2
                        cuda.syncthreads()

                    ### write the block's scan results to global
                    if ii_ind < end:
                        save(src_vals, ii_src, my_val)

                    ### save the carry
                    if pos == 0:
                        carry_val = load(shared_vals, threads_per_block - 1)
                        carry_is_root = shared_is_root[threads_per_block - 1]

                    cuda.syncthreads()

        self.threads_per_block = threads_per_block
        self._segscan_by_generation = _segscan_by_generation
