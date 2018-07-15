import math
import numba
import numba.cuda as cuda


class GenerationalSegmentedScan:
    """Factory class for cuda-based generational segmented scan operations.

    GenerationalSegmentedScan provides a factory class managing creation and
    invocation of generic, generational segement scan operations.

    # Scan Overview

    The scan operation processes linear scan paths with an associative binary
    operator, where the scan paths many have any number of additional off-path
    node inputs added *before* the scan path. For example, consider the
    operation composed of path values (P), off path values (OP) joined by an
    operator (+):

        OP_0            OP_1
          +               +
          |               |
          v               v
        P_0+--->P_1+--->P_2+--->P_3+--->P_4
                          ^
                          |
                          +
                        OP_2

    Represents the complete operation:

      (OP_0 + P_0) + P_1 + (OP_1 + OP_2 + P_2) + P_3 + P_4

    As this is a scan, rather than reduction, this results in:

        R_0-----R_1-----R_2-----R_3-----R_4

        R_0 = OP_0 + P_0
        R_1 = (OP_0 + P_0) + P_1
        R_2 = (OP_0 + P_0) + P_1 + (OP_1 + OP_2 + P_2)
        R_3 = (OP_0 + P_0) + P_1 + (OP_1 + OP_2 + P_2) + P_3
        R_4 = (OP_0 + P_0) + P_1 + (OP_1 + OP_2 + P_2) + P_3 + P_4

    The off-path inputs of a scan are taken from the result values of previous
    scan operations. Scans are processed by "generation", arranged such that
    the off-path inputs of any segment are draw *exclusively* from earlier
    generations. Scans within a generation are processed via a parallel
    segmented scan.

    # Implementation

    Implementation of a specific scan operation involves subclassing this
    template class and providing:

        `@cuda.jit(device=True)` class-methods:

            add: ((val, val) -> val)
                The associative operator.
            zero : (() -> val)
                The identity value for the associative operator.
            load: (array, i) -> val
                Load val as a local value from an array of values.
            save: (array, i, val) -> ()
                Save val from a local value to an array of values.

        class properties:
            val_shape: tuple[int, ...]
                The sub-shape of val in the arrays referenced by 'load'/'save'.

    The subclass may then be used as singleton provider for scan invocations
    via the class method `segscan_by_generation`. Kernel functions are
    specialized for specific thread_per_block values, with kernels requiring a
    float64 shared memory allocation of shape (threads_per_block,) + val_shape.
    """

    @classmethod
    def segscan_by_generation(
            cls,
            threads_per_block,  # int
            src_vals,  # [n] + [val_shape]
            scan_to_src_ordering,  # [n]
            is_path_root,  #[n]
            non_path_inputs,  #[n, max_num_inputs]
            generation_ranges,  #[g, 2]
    ):
        cls.get_kernel(threads_per_block)[1, threads_per_block](
            src_vals,
            scan_to_src_ordering,
            is_path_root,
            non_path_inputs,
            generation_ranges,
        )

    @classmethod
    def get_kernel(cls, threads_per_block):

        # Use __dict__ (not hasattr) to only search cls (not superclass).
        if "_kernel_cache" not in cls.__dict__:
            setattr(cls, "_kernel_cache", dict())

        _kernel_cache = getattr(cls, "_kernel_cache")

        if threads_per_block not in _kernel_cache:
            _kernel_cache[threads_per_block
                          ] = cls._generate_kernel(threads_per_block)

        return _kernel_cache[threads_per_block]

    @classmethod
    def _generate_kernel(cls, threads_per_block):
        if not math.log2(threads_per_block).is_integer():
            raise ValueError(
                f"thread_per_block must be power of 2: {threads_per_block}"
            )

        ### Initialize numba kernel via lexical closure.

        # The subclass interface must be available as variables in the
        # jit-function closure defined below, rather than accessed via self.*,
        # so that numba.jit can properly bind the device functions as jit-time.
        add = cls.add
        zero = cls.zero
        load = cls.load
        save = cls.save

        shared_shape = (threads_per_block, ) + cls.val_shape

        n_scan_iter = int(math.log2(threads_per_block))

        # cuda.jit a dynamically defined function so that the variables defined
        # here, in the lexical scope of the function definition, are bound and
        # made available to numba.
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

        return _segscan_by_generation
