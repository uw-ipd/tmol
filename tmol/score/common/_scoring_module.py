import torch

# from tmol.score.hbond.potentials import hbond_pose_scores
from tmol.score.common import convert_float64


class _ZeroCoordinates(torch.autograd.Function):
    """Attach a no-op coordinate gradient without launching a device kernel."""

    @staticmethod
    def forward(ctx, coords, zero, shape):
        return zero.expand(shape)

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None


class ZeroTermPoseScoringModule(torch.nn.Module):
    """A zero-cost placeholder for a score term absent from a pose topology.

    The scalar buffers expand to the requested output shape without allocating
    or launching a device kernel. Keeping the module in the score-function
    output preserves score-type ordering for callers requesting per-term
    energies.
    """

    def __init__(self, classname, n_score_types, n_poses, device, max_n_blocks=None):
        super().__init__()
        self.classname = classname
        shape = [n_score_types, n_poses]
        if max_n_blocks is not None:
            shape.extend([max_n_blocks, max_n_blocks])
        self.shape = tuple(shape)
        self.register_buffer(
            "_zero_f32", torch.zeros((), dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "_zero_f64", torch.zeros((), dtype=torch.float64, device=device)
        )

    def forward(self, coords):
        zero = self._zero_f64 if coords.dtype == torch.float64 else self._zero_f32
        # Some callers benchmark or inspect a single energy term by calling
        # backward() even when that term has no coordinate dependence. Attach
        # a custom no-op edge so that convention remains valid without a
        # reduction kernel or a persistent leaf gradient.
        if torch.is_grad_enabled() and coords.requires_grad:
            return _ZeroCoordinates.apply(coords, zero, self.shape)
        return zero.expand(self.shape)


class TermScoringModule(torch.nn.Module):
    def __init__(
        self,
        classname,
        term_parameters,
        term_score_poses,
    ):
        super(TermScoringModule, self).__init__()
        self.classname = classname
        self.term_parameters = []

        self.add_parameters(self.term_parameters, term_parameters)

        self.term_score_poses = term_score_poses

    def _build_static_tails(self, block_pair_scoring):
        # Pre-build the non-coordinate argument list used by every forward.
        # Float64 scoring is mainly used by gradcheck, so defer those copies
        # until a float64 input actually arrives.
        tail = list(self.common_parameters) + list(self.term_parameters)
        self._static_tail_f32 = tuple(tail) + (block_pair_scoring,)
        self._static_tail_f64 = None

    def _static_tail_for_coords(self, coords):
        if coords.dtype != torch.float64:
            return self._static_tail_f32
        if self._static_tail_f64 is None:
            tail_f64 = list(self._static_tail_f32[:-1])
            convert_float64(tail_f64)
            self._static_tail_f64 = tuple(tail_f64) + (self._static_tail_f32[-1],)
        return self._static_tail_f64

    def add_parameters(self, table, params):
        def _p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        for param in params:
            table += [_p(param) if type(param) is torch.Tensor else param]


class TermPoseScoringModule(TermScoringModule):
    def __init__(
        self,
        classname,
        pose_stack,
        term_parameters,
        term_score_poses,
    ):
        super(TermPoseScoringModule, self).__init__(
            classname, term_parameters, term_score_poses
        )

        self.common_parameters = []

        self.add_parameters(
            self.common_parameters,
            [
                pose_stack.rot_coord_offset,
                pose_stack.pose_ind_for_atom,
                pose_stack.first_rot_for_block,
                pose_stack.block_type_ind,  # block_type for first rot for block
                pose_stack.block_ind_for_rot,
                pose_stack.pose_ind_for_rot,
                pose_stack.block_type_ind_for_rot,
                pose_stack.n_rots_for_pose,
                pose_stack.rot_offset_for_pose,
                pose_stack.n_rots_for_block,
                pose_stack.rot_offset_for_block,
                pose_stack.max_n_rots_per_pose,
            ],
        )
        self.n_poses = pose_stack.first_rot_for_block.shape[0]
        self.max_n_blocks = pose_stack.first_rot_for_block.shape[1]


class TermWholePoseScoringModule(TermPoseScoringModule):
    def __init__(
        self,
        classname,
        pose_stack,
        term_parameters,
        term_score_poses,
    ):
        super(TermWholePoseScoringModule, self).__init__(
            classname, pose_stack, term_parameters, term_score_poses
        )
        self.count = 0
        self._build_static_tails(False)

    def forward(
        self,
        coords,
    ):
        flat = coords.flatten(start_dim=0, end_dim=-2)
        tail = self._static_tail_for_coords(coords)
        # ignore the dispatch_indices return tensor
        scores, _ = self.term_score_poses(flat, *tail)
        return scores


class TermBlockPairScoringModule(TermPoseScoringModule):
    def __init__(
        self,
        classname,
        pose_stack,
        term_parameters,
        term_score_poses,
    ):
        super(TermBlockPairScoringModule, self).__init__(
            classname, pose_stack, term_parameters, term_score_poses
        )
        self._build_static_tails(True)

    def forward(
        self,
        coords,
    ):
        flat = coords.flatten(start_dim=0, end_dim=-2)
        tail = self._static_tail_for_coords(coords)
        scores, _ = self.term_score_poses(flat, *tail)
        return scores


class TermRotamerScoringModule(TermScoringModule):
    def __init__(
        self,
        classname,
        rotamer_set,
        term_parameters,
        term_score_poses,
    ):
        super(TermRotamerScoringModule, self).__init__(
            classname, term_parameters, term_score_poses
        )

        self.common_parameters = []

        def _i32(x):
            return x if isinstance(x, int) else x.to(torch.int32)

        self.add_parameters(
            self.common_parameters,
            [
                _i32(t)
                for t in [
                    rotamer_set.coord_offset_for_rot,  # rot coord offset
                    rotamer_set.pose_ind_for_atom,  # pose_ind_for_atom?? unused
                    rotamer_set.rot_offset_for_block,  # first rot for block
                    rotamer_set.first_rot_block_type,  # first rot block type
                    rotamer_set.block_ind_for_rot,
                    rotamer_set.pose_for_rot,
                    rotamer_set.block_type_ind_for_rot,
                    rotamer_set.n_rots_for_pose,
                    rotamer_set.rot_offset_for_pose,
                    rotamer_set.n_rots_for_block,
                    rotamer_set.rot_offset_for_block,  # three times?!
                    rotamer_set.max_n_rots_per_pose,
                ]
            ],
        )
        self.n_poses = rotamer_set.n_rots_for_pose.shape[0]
        self.n_rots = rotamer_set.coord_offset_for_rot.shape[0]
        self._build_static_tails(True)

    def forward(self, coords):
        """Return (scores, indices) without creating any sparse tensor.

        scores:  [n_subterms, nnz] float32
        indices: [3, nnz]          int32  (pose, rot_i, rot_j in global rot numbering)
        """
        flat = coords.flatten(start_dim=0, end_dim=-2)
        tail = self._static_tail_for_coords(coords)
        scores, indices = self.term_score_poses(flat, *tail)
        return scores, indices

    def forward_split(self, coords):
        scores, indices = self.forward(coords)
        # Legacy path: kept for any callers outside RotamerScoringModule.
        # Prefer forward + caller-side weighting to avoid the torch.stack
        # memory cost (stacking creates a [n_subterms, n_poses, n_rots, n_rots]
        # 4D sparse tensor whose index storage is n_subterms x nnz x 4 int32).
        return torch.stack(
            [
                torch.sparse_coo_tensor(
                    indices,
                    scores[subterm, :],
                    size=(self.n_poses, self.n_rots, self.n_rots),
                )
                for subterm in range(scores.size(0))
            ]
        )
