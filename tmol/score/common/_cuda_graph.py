"""CUDA scoring capture with ownership that follows its pending backwards."""

import torch


class _ReplayScoringGraph(torch.autograd.Function):
    # A module-level Function keeps per-capture tensors out of the reference
    # cycle between dynamically created forward and backward Function classes.
    @staticmethod
    def forward(ctx, capture, coords, *parameters):
        ctx.capture = capture
        if capture._coords.data_ptr() != coords.data_ptr():
            capture._coords.copy_(coords)
        capture._forward_graph.replay()
        # The stored capture output must never acquire this Function's context:
        # that would create a new cycle through ctx.capture.
        return capture._output.detach()

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        capture = ctx.capture
        if capture._backward_graph is None:
            return (None,) * (2 + len(capture._parameters_for_capture))
        if capture._grad_output.data_ptr() != grad_output.data_ptr():
            capture._grad_output.copy_(grad_output)
        capture._backward_graph.replay()
        return (None,) + tuple(
            grad.detach() if grad is not None else None for grad in capture._grad_inputs
        )


class CapturedScoringGraph(torch.nn.Module):
    """Capture one coordinate-input, tensor-output scoring module.

    The capture owns the unmodified module so native kernels retain all their
    parameter storage, including tensors held in ordinary Python containers.
    Outstanding autograd outputs retain the capture until backward is finished
    and the outputs are released. No cyclic garbage collection is required.
    """

    def __init__(self, module: torch.nn.Module, example_coords: torch.Tensor):
        super().__init__()
        if torch.is_autocast_enabled() and torch.is_autocast_cache_enabled():
            raise RuntimeError("Captured scoring requires autocast cache_enabled=False")
        self.module = module
        self.training = module.training
        self._graph_training_state = module.training
        self._coords = example_coords
        self._parameters_for_capture = tuple(module.parameters())
        inputs = (example_coords,) + self._parameters_for_capture
        differentiable = tuple(value for value in inputs if value.requires_grad)

        with torch.cuda.device(example_coords.device), torch.enable_grad():
            torch.cuda.synchronize()
            # Torch's default graph-capture stream may belong to another
            # device. Warm and capture on the same explicit device stream.
            stream = torch.cuda.Stream(device=example_coords.device)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    output = module(example_coords)
                    if output.requires_grad:
                        gradients = torch.autograd.grad(
                            output,
                            differentiable,
                            torch.empty_like(output),
                            allow_unused=True,
                        )
                        del gradients
                    del output
            torch.cuda.synchronize()

            # Forward and backward run in this order and can share a pool.
            pool = torch.cuda.graph_pool_handle()
            self._forward_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._forward_graph, pool=pool, stream=stream):
                self._output = module(example_coords)

            self._backward_graph = None
            gradients = ()
            if self._output.requires_grad:
                self._grad_output = torch.empty_like(self._output)
                self._backward_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._backward_graph, pool=pool, stream=stream):
                    gradients = torch.autograd.grad(
                        self._output,
                        differentiable,
                        self._grad_output,
                        allow_unused=True,
                    )
            gradient_iter = iter(gradients)
            self._grad_inputs = tuple(
                next(gradient_iter) if value.requires_grad and gradients else None
                for value in inputs
            )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if self.module.training != self._graph_training_state:
            return self.module(coords)
        return _ReplayScoringGraph.apply(self, coords, *self._parameters_for_capture)
