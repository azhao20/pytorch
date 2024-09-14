# Owner(s): ["module: unknown"]
import math
import os
import pickle
import numpy as np
from collections import defaultdict
from typing import Any, Callable, Dict, List, Set, Tuple
from typing_extensions import Self

import torch
import torch.utils._pytree as pytree
from torch._guards import active_fake_mode
from torch._inductor.utils import get_device_tflops, get_gpu_dram_gbps
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.mod_tracker import ModTracker
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import flop_registry


aten = torch.ops.aten

# This value is hard-coded here:
# https://github.com/pytorch/pytorch/blob/5fba5d83f0703ff8077ab65448a998e9ad6598fd/c10/cuda/CUDACachingAllocator.cpp#L117
_PYTORCH_MIN_ALLOCATE = (
    2**9 if int(os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING", 0)) == 0 else 1
)

# No fall-back kernel needed/exists for view ops
_VIEW_OPS = {
    aten.lift_fresh,
    aten.t,
    aten.transpose,
    aten.view,
    aten.detach,
    aten._unsafe_view,
    aten.split,
    aten.adjoint,
    aten.as_strided,
    aten.diagonal,
    aten.expand,
    aten.expand_as,
    aten.movedim,
    aten.permute,
    aten.select,
    aten.squeeze,
    aten.mT,
    aten.mH,
    aten.real,
    aten.imag,
    aten.view_as,
    aten.unflatten,
    aten.unfold,
    aten.unbind,
    aten.unsqueeze,
    aten.vsplit,
    aten.hsplit,
    aten.split_with_sizes,
    aten.swapaxes,
    aten.swapdims,
    aten.chunk,
}
# We can ignore benchmarking tensor create ops
_CREATE_OPS = {
    aten.randint,
    aten.randn,
    aten.rand,
    aten.randn_like,
    aten.rand_like,
    aten.randint_like,
    aten.arange,
    aten.ones_like,
    aten.zeros_like,
}

_IGNORE_OPS = _VIEW_OPS | _CREATE_OPS

# Similar to `flop_registry`, stores the functions that make predictions
_LEARNED_OPS: Dict[Any, Any] = {}

# Caches the learned models that predict ops' runtimes.
_LEARNED_OPS_PREDICTORS: Dict[str, Any] = {}

_LEARNED_ROOFLINE_OPS: Dict[Any, Any] = {}


__all__ = ["RuntimeEstimator"]


class RuntimeEstimator(TorchDispatchMode):
    """
    Estimates the GPU runtime in milliseconds using various estimation methods under the ``FakeTensorMode``.

    This class provides a ``TorchDispatchMode`` based context manager that can be used to estimate the eager
    runtime of PyTorch functions. It supports two estimation modes, benchmarking (`operator-level-benchmark`) and
    roofline cost modeling (`operator-level-cost-model`).
    For modules executed under this context manager, it agggregates the forward and backward operation runtimes
    and also records their execution orders.

    Attributes:
        mod_runtimes (Dict[str, Dict[str, float]]): A dictionary of module runtimes. The key to the outer dictionary
            is the fully qualified name (FQN) of the module. For each module the forward and backward runtimes of the
            operations are aggregated in the inner dictionary keyed by 'fw' and 'bw'.
        mod_fw_pre_order (List[str]): List of module FQNs in pre-forward execution order.
        mod_bw_pre_order (List[str]): List of module FQNs in pre-backward execution order.
        mod_fw_post_order (List[str]): List of module FQNs in post-forward execution order.
        mod_bw_post_order (List[str]): List of module FQNs in post-backward execution order.
        total_runtime (float): The total estimated runtime in milliseconds.

    Note:
        1) The benchmarking estimate mode will execute kernels on GPU and assumes that every operation can run in
            isolation without causing an OOM error. It is also designed to be used only under ``FakeTensorMode``.
        2) Currently wrapper tensor sub-classes such as ``DTensor`` won't produce correct estimates. We plan to support
            them in future PRs.
        3) We only estimate the compute time, if your code has communication, it will not be considered. Again, we will
            support this in future PRs.

    Example usage:

        .. code-block:: python

            runtime_estimator = RuntimeEstimator()
            with FakeTensorMode():
                module = ...
                optimizer = ...
                inp = ...
                with runtime_estimator(estimate_mode_type="operator-level-cost-model"):
                    loss = module(inp)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                runtime_estimator.display_modulewise_stats()
    """

    _float_types: Set[torch.dtype] = {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }
    _no_fallback_kernel: Set[torch._ops._OpNamespace] = set()
    fake_mode: FakeTensorMode

    def __init__(self) -> None:
        super().__init__()
        self._estimate: Callable
        self._estimate_mode_type: str
        self._mod_tracker = ModTracker()
        self.mod_runtimes: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(lambda: 0.0)
        )
        self.mod_fw_pre_order: List[str] = []
        self.mod_bw_pre_order: List[str] = []
        self.mod_fw_post_order: List[str] = []
        self.mod_bw_post_order: List[str] = []
        self.total_runtime: float = 0.0

    # Adapted from: https://github.com/pytorch/pytorch/blob/9b902b3ee3bd608a19543362b66bf06c373dd374/torch/_subclasses/fake_tensor.py#L1969  # noqa: PGH004,B950
    # NB: returns fake tensors
    @classmethod
    def _maybe_run_and_benchmark_fallback_kernel(  # type: ignore[no-untyped-def]
        cls,
        func,
        args,
        kwargs,
        orig_not_implemented_exception,
    ):
        """
        Runs and benchmarks a fallback kernel for a given function.

        Args:
            func (Callable): The function to benchmark.
            args (Tuple): The arguments to pass to the function.
            kwargs (Dict[str, Any]): The keyword arguments to pass to the function.
            orig_not_implemented_exception (Exception): The original exception to raise if the fallback kernel
                is not implemented.

        Returns:
            Tuple[Any, float]: A tuple containing the result of the function and
                the mean operation time in milliseconds.
        """
        # these should all be supported, just to be safe
        # avoid fallback for operators which inplace modify metadata
        # because the input fake tensors would be umodified
        if torch.Tag.inplace_view in func.tags:  # type: ignore[attr-defined]
            raise orig_not_implemented_exception

        inp_impls = {}
        flat_args, args_spec = pytree.tree_flatten((args, kwargs))
        # Don't use in_kernel_invocation_manager(fake_mode) as we want to do
        # REAL compute (not with meta device)
        with no_dispatch():

            def to_real_tensor(e):  # type: ignore[no-untyped-def]
                if cls.fake_mode.is_our_fake(e):
                    if e.dtype in cls._float_types:
                        out = torch.rand_like(e, device=e.fake_device)
                    else:
                        out = torch.ones_like(e, device=e.fake_device)
                    if e.is_sparse:
                        out._coalesced_(e.is_coalesced())
                    inp_impls[id(out)] = e
                    return out
                return e

            flat_args = [to_real_tensor(a) for a in flat_args]
            args, kwargs = pytree.tree_unflatten(flat_args, args_spec)
            r = func(*args, **kwargs)
            warmup_iters, actual_iters = 2, 3
            for _ in range(warmup_iters):
                func(*args, **kwargs)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(torch.cuda.current_stream())
            for _ in range(actual_iters):
                func(*args, **kwargs)
            end_event.record(torch.cuda.current_stream())
            torch.cuda.synchronize()
            cuda_time = start_event.elapsed_time(end_event)
            mean_op_time = cuda_time / actual_iters

        storages = set()

        for e in flat_args:
            if isinstance(e, torch.Tensor):
                if not e.is_sparse:
                    storages.add(e._typed_storage()._cdata)

        # TODO: also check metadata change on inputs
        # proper aliasing/metadata relationship between outputs and inputs will
        # not be set up, bc of conversion to device, unless we can reuse an
        # input impl

        def map_out(e):  # type: ignore[no-untyped-def]
            if id(e) not in inp_impls and (
                isinstance(e, torch.Tensor)
                and not e.is_sparse
                and e._typed_storage()._cdata in storages
            ):
                raise orig_not_implemented_exception

            if isinstance(e, torch.Tensor):
                if id(e) in inp_impls:
                    return inp_impls[id(e)]
                else:
                    return cls.fake_mode.fake_tensor_converter.from_real_tensor(
                        cls.fake_mode, e
                    )
            else:
                return e

        return (pytree.tree_map(map_out, r), mean_op_time)

    @classmethod
    def _benchmark_estimate(cls, func, args, kwargs) -> Tuple[Any, float]:  # type: ignore[no-untyped-def]
        """
        Estimates the runtime of a function using benchmarking.

        Args:
            func: The function to estimate.
            args: The arguments to pass to the function.
            kwargs: The keyword arguments to pass to the function.
            res: The result of the function.

        Returns:
            Tuple[Any, float]: A tuple containing the result of the function and
                the mean operation time in milliseconds.
        """
        assert isinstance(
            cls.fake_mode, FakeTensorMode
        ), "Initialize/Assign FakeTensorMode before using this function"
        mean_op_time = 0.0
        if func._overloadpacket not in _VIEW_OPS:
            try:
                res, mean_op_time = cls._maybe_run_and_benchmark_fallback_kernel(
                    func,
                    args,
                    kwargs,
                    NotImplementedError,
                )
                return (res, mean_op_time)
            except NotImplementedError:
                cls._no_fallback_kernel.add(func._overloadpacket)
        res = func(*args, **kwargs or {})
        return (res, mean_op_time)

    @classmethod
    def _get_transfer_time(flat_args_kwargs, flat_outs) -> float:  # type: ignore[no-untyped-def]
        """
        Estimates the memory transfer time of input and output tensors.

        Args:
            flat_args_kwargs (List[torch.Tensor]): The flat list of arguments and keyword arguments.
            flat_outs (List[torch.Tensor]): The flat list of outputs.

        Returns:
            float: The estimated memory transfer time in nanoseconds.
        """
        def get_num_bytes(t: torch.Tensor) -> int:
            """
            Calculates the memory consumption of a tensor.

            Args:
                t (torch.Tensor): The input tensor.

            Returns:
                int: The memory consumption of the tensor in bytes.
            """
            num_bytes = t.untyped_storage().nbytes()
            mem_consumed = (
                math.ceil(num_bytes / _PYTORCH_MIN_ALLOCATE) * _PYTORCH_MIN_ALLOCATE
            )
            return mem_consumed

        gpu_memory_bandwidth = get_gpu_dram_gbps()
        read_bytes = sum(
            get_num_bytes(t)
            for t in flat_args_kwargs
            if isinstance(t, torch.Tensor)
        )
        write_bytes = sum(
            get_num_bytes(t) for t in flat_outs if isinstance(t, torch.Tensor)
        )
        counted_bytes = read_bytes + write_bytes
        # The GPU memory bandwidth is in GB/s so the transfer time is in nanoseconds
        transfer_time = counted_bytes / gpu_memory_bandwidth
        return transfer_time


    # Adapted from: https://github.com/pytorch/pytorch/blob/9b902b3ee3bd608a19543362b66bf06c373dd374/torch/_inductor/scheduler.py#L589  # noqa: PGH004,B950
    @classmethod
    def _roofline_estimate(cls, func, args, kwargs) -> Tuple[Any, float]:  # type: ignore[no-untyped-def]
        """
        Estimates the runtime of a function using a roofline cost model.

        Args:
            func: The function to estimate.
            args: The arguments to pass to the function.
            kwargs: The keyword arguments to pass to the function.
            out: The output of the function.

        Returns:
            Tuple[Any, float]: A tuple containing the result of the function and
                the mean operation time in milliseconds.
        """
        assert (
            torch.cuda.is_available()
        ), "Roofline estimation needs to access CUDA capabilities to make estimations"

        def get_compute_time(func_packet, args, kwargs, out, out_dtypes) -> float:  # type: ignore[no-untyped-def]
            """
            Estimates the compute time of an aten operator.

            Args:
                func_packet: The operator overload packet.
                args: The arguments to the operator.
                kwargs: The keyword arguments to the operator.
                out: The output of the operator.
                out_dtypes: The output data types.

            Returns:
                float: The estimated compute time in nanoseconds.
            """
            if func_packet in flop_registry:
                assert (
                    len(out_dtypes) == 1
                ), f"Only support single out dtype got {out_dtypes} for {func_packet}"
                dtype = out_dtypes.pop()
                # This actually gives peta-FLOPs/s hence multiply by 1e15 to get the FLOPs/s
                peak_gpu_flops = get_device_tflops(dtype) * 1e15
                # We can expect to achieve 75% of theoretical peak flops
                factor = 0.75
                peak_empirical_flops = factor * peak_gpu_flops
                flop_count_func = flop_registry[func_packet]
                # We divide by a factor of 2 to get the MACs (multiply and accumulate)
                flop_count = flop_count_func(*args, **kwargs, out_val=out) / 2
                # We multiply by 1e9 to get the time in nano seconds
                compute_time = (flop_count / peak_empirical_flops) * 1e9
                return compute_time
            return 0.0

        # Roofline Cost Model Explanation

        # The roofline cost model estimates the execution time of an operator based on
        # the device's empirical maximum FLOPs/sec (pi) and device DRAM bandwidth (beta).

        # Variables:
        # - pi: Maximum empirical FLOPs/sec of the device
        # - beta: Maximum empirical device DRAM bandwidth (bytes/sec) of the device
        # - I: Arithmetic intensity of the operator (FLOPs/bytes)
        # - op_flops: FLOPs required by the operator
        # - op_bytes: Bytes transferred to and from DRAM for the operator

        # Calculation Steps:
        # 1. Calculate arithmetic intensity: I = op_flops / op_bytes
        # 2. Calculate estimated FLOPs/sec: est_flops_sec = min(pi, beta * I)
        # 3. Calculate estimated operator time: estimated_op_time = op_flops / est_flops_sec
        #    This simplifies to: estimated_op_time = max(op_flops / pi, op_flops / (beta * I))
        #    Further simplifying: estimated_op_time = max(op_flops / pi, op_bytes / beta)

        # Simplified Formulas:
        # - compute_time = op_flops / pi
        # - transfer_time = op_bytes / beta
        # - estimated_op_time = max(compute_time, transfer_time)

        kwargs = kwargs if kwargs else {}
        out = func(*args, **kwargs)
        op_time = 0.0
        func_packet = func._overloadpacket
        if func_packet not in _IGNORE_OPS:
            flat_args_kwargs, args_spec = pytree.tree_flatten((args, kwargs))
            flat_outs, out_spec = pytree.tree_flatten(out)
            transfer_time = cls._get_transfer_time(flat_args_kwargs, flat_outs)

            out_dtypes = {
                t.dtype
                for t in flat_outs
                if isinstance(t, torch.Tensor) and t.dtype in cls._float_types
            }

            args, kwargs = pytree.tree_unflatten(flat_args_kwargs, args_spec)
            out = pytree.tree_unflatten(flat_outs, out_spec)

            compute_time = get_compute_time(func_packet, args, kwargs, out, out_dtypes)
            # We get the estimated time as the max of the transfer time and
            # compute time. We divide by 1e6 to get the time in ms
            op_time = max(transfer_time, compute_time) / 1e6

        return (out, op_time)
        
    @classmethod
    def _learned_estimate_predictor(func_packet, args, kwargs, out, out_dtypes) -> float:  # type: ignore[no-untyped-def]
        """
        TODO:
            1) the order of the features
            2) where the models are stored
        
        
        Estimates the compute time of an aten operator.

        Args:
            func_packet: The operator overload packet.
            args: The arguments to the operator.
            kwargs: The keyword arguments to the operator.
            out: The output of the operator.
            out_dtypes: The output data types.

        Returns:
            float: The estimated compute time in nanoseconds.
            
        
        # TODO: comments.
        Note: for the prediction functions, we mimic the arguments for mm_flop.
        """
        def get_learned_model(op: str) -> Any:
            """
            Expects predictor to be stored as <op>_predictor.pkl
            """
            if op not in _LEARNED_OPS_PREDICTORS:
                
                base_dir = os.path.join(os.getcwd())
                path = os.path.join(base_dir, op, "_predictor.pkl")
                
                _LEARNED_OPS_PREDICTORS[op] = pickle.load(path)
            return _LEARNED_OPS_PREDICTORS[op]
        
        
        from functools import wraps
        from torch.utils._pytree import tree_map
        
        def get_shape(i):
            if isinstance(i, torch.Tensor):
                return i.shape
            return i

        def shape_wrapper(f):
            """
            Similar to flop_counter.shape_wrapper(), but also takes takes gflops.
            """
            @wraps(f)
            def nf(dtype, gflops, *args, out_val=None, **kwargs):
                args, kwargs, out_shape = tree_map(get_shape, (args, kwargs, out_val))
                return f(dtype, gflops, *args, out_shape=out_shape, **kwargs)
            return nf

        def register_timing_formula(targets, get_raw=False):
            """
            Similar to flop_counter.register_flop_formula().
            """
            def register_fun(flop_formula):
                if not get_raw:
                    flop_formula = shape_wrapper(flop_formula)

                def register(target):
                    if not isinstance(target, torch._ops.OpOverloadPacket):
                        raise ValueError(
                            f"register_flop_formula(targets): expected each target to be "
                            f"OpOverloadPacket (i.e. torch.ops.mylib.foo), got "
                            f"{target} which is of type {type(target)}")
                    if target in flop_registry:
                        raise RuntimeError(f"duplicate registrations for {target}")
                    flop_registry[target] = flop_formula

                # To handle allowing multiple aten_ops at once
                torch.utils._pytree.tree_map_(register, targets)

                return flop_formula

            return register_fun
        
        
        def convert_dtype(dtype) -> list[int]:
            """
            To use dtype in a learned model, we convert them to one-hot encodings.
            
            Learned model supports the dtypes:
                - torch.float16
                - torch.float32
                - torch.bfloat16
            """
            dtypes = [torch.float16, torch.float32, torch.bfloat16]
            return [1 if dtype == d else 0 for d in dtypes]
        
        @register_timing_formula([aten.mm, aten.addmm])
        def mm_time(dtype, gflops, a_shape, b_shape, *args, out_shape=None, **kwargs) -> float:
            model = get_learned_model("mm")
            
            m, n = a_shape
            n2, p = b_shape
            assert n == n2
            
            features = np.array([[m, n, p, gflops] + convert_dtype(dtype)])
            return model.predict(features)
            
        @register_timing_formula([aten.bmm, aten.baddmm])
        def bmm_time(dtype, gflops, a_shape, b_shape, out_shape=None, **kwargs) -> float:
            model = get_learned_model("bmm")
            
            b, m, n = a_shape
            b2, n2, p = b_shape
            assert b == b2 and n == n2
            
            features = np.array([[b, m, n, p, gflops] + convert_dtype(dtype)])
            return model.predict(features)

        @register_timing_formula(aten._scaled_dot_product_efficient_attention)
        def sdpa_efficient_time(dtype, gflops, query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs) -> float:
            """
            TODO: is_causal. How to track?
            """
            model = get_learned_model("sdpa")
            
            b, h, s_q, d_qk = query_shape
            _b2, _h2, s_kv, _d2 = key_shape
            _b3, _h3, _s3, d_v = value_shape
            assert b == _b2 == _b3 and h == _h2 == _h3 and d_qk == _d2 and s_kv == _s3 and d_qk == _d2

            backends_ohe = [1, 0]
            is_causal_ohe = [1, 0]
            features = np.array([[b, h, s_q, s_kv, d_qk, d_v, gflops] + convert_dtype(dtype) + backends_ohe + is_causal_ohe])
            return model.predict(features)
        
        @register_timing_formula(aten._scaled_dot_product_flash_attention)
        def sdpa_flash_time(dtype, gflops, query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs) -> float:
            model = get_learned_model("sdpa")
            
            b, h, s_q, d_qk = query_shape
            _b2, _h2, s_kv, _d2 = key_shape
            _b3, _h3, _s3, d_v = value_shape
            assert b == _b2 == _b3 and h == _h2 == _h3 and d_qk == _d2 and s_kv == _s3 and d_qk == _d2

            backends_ohe = [0, 1]
            is_causal_ohe = [1, 0]
            features = np.array([[b, h, s_q, s_kv, d_qk, d_v, gflops] + convert_dtype(dtype) + backends_ohe + is_causal_ohe])
            return model.predict(features)

        # @register_timing_formula
        # def sdpa_backward_time(dtype, gflops, grad_out_shape, query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs) -> float:
        #     model = get_learned_model("sdpa_backward")
            
        #     b, h, s_q, d_q = query_shape
        #     _b2, _h2, s_k, _d2 = key_shape
        #     _b3, _h3, _s3, d_v = value_shape
        #     _b4, _h4, _s4, _d4 = grad_out_shape
        #     assert b == _b2 == _b3 == _b4 and h == _h2 == _h3 == _h4 and d_q == _d2
        #     assert d_v == _d4 and s_k == _s3 and s_q == _s4
            
        #     # features = np.array([[b, m, k, n, gflops]])
        #     return model.predict(features)

        @register_timing_formula([aten.convolution, aten._convolution])
        def conv_time(dtype, gflops, x_shape, w_shape, _bias, _stride, _padding, _dilation, transposed, *args, out_shape=None, **kwargs) -> float:
            """
            TODO: need to add support for higher dims.
            """
            model = get_learned_model("conv")
            
            # batch_size = x_shape[0]
            # conv_shape = (x_shape if transposed else out_shape)[2:]
            # c_out, c_in, *filter_size = w_shape
            
            # features = np.array([[b, m, k, n, gflops]])
            return model.predict(features)
        
        @register_timing_formula(aten.convolution_backward)
        def conv_backward_time(
            dtype,
            gflops,
            grad_out_shape,
            x_shape,
            w_shape,
            _bias,
            _stride,
            _padding,
            _dilation,
            transposed,
            _output_padding,
            _groups,
            output_mask,
            out_shape) -> float:
            """
            TODO: need to add support for higher dims.
            
            Also: don't support...
            """
            model = get_learned_model("conv_backward")
            
            # features = np.array([[b, m, k, n, gflops]])
            return model.predict(features)
        
        
        if func_packet in _LEARNED_OPS:
            assert (
                len(out_dtypes) == 1
            ), f"Only support single out dtype got {out_dtypes} for {func_packet}"
            dtype = out_dtypes.pop()
            
            flop_count_func = flop_registry[func_packet]
            gflops = flop_count_func(*args, **kwargs, out_val=out) / 1e9
            
            predictor_func = _LEARNED_OPS[func_packet]
            # Returns compute time in ms, so multiply by 1e6 to get nanoseconds
            compute_time = predictor_func(dtype, gflops, *args, **kwargs, out_val=out)
            compute_time *= 1e6
            return compute_time
        return 0.0
    
    @classmethod
    def _learned_estimate(cls, func, args, kwargs) -> Tuple[Any, float]:  # type: ignore[no-untyped-def]
        """
        TODO: add docstring
        - We use one model per.
        - Maybe: map operators to functions
        
        
        Estimates the runtime of a function using benchmarking.

        Args:
            func: The function to estimate.
            args: The arguments to pass to the function.
            kwargs: The keyword arguments to pass to the function.
            res: The result of the function.

        Returns:
            Tuple[Any, float]: A tuple containing the result of the function and
                the mean operation time in milliseconds.
        """
        assert (
            torch.cuda.is_available()
        ), "Roofline estimation needs to access CUDA capabilities to make estimations"
        
        kwargs = kwargs if kwargs else {}
        out = func(*args, **kwargs)
        op_time = 0.0
        func_packet = func._overloadpacket
        if func_packet not in _IGNORE_OPS:
            flat_args_kwargs, args_spec = pytree.tree_flatten((args, kwargs))
            flat_outs, out_spec = pytree.tree_flatten(out)
            transfer_time = cls._get_transfer_time(flat_args_kwargs, flat_outs)

            out_dtypes = {
                t.dtype
                for t in flat_outs
                if isinstance(t, torch.Tensor) and t.dtype in cls._float_types
            }

            args, kwargs = pytree.tree_unflatten(flat_args_kwargs, args_spec)
            out = pytree.tree_unflatten(flat_outs, out_spec)
            
            compute_time = cls._learned_estimate_predictor(func_packet, args, kwargs, out, out_dtypes)
            # We get the estimated time as the max of the transfer time and
            # compute time. We divide by 1e6 to get the time in ms
            op_time = max(transfer_time, compute_time) / 1e6
            
        return (out, op_time)

    def display_modulewise_stats(self, depth: int = 2) -> None:
        """
        Displays module-wise statistics collected by ``RuntimeEstimator``.

        Prints the pre-forward and pre-backward execution orders.
        Displays the module-wise forward and backward runtimes in milliseconds.

        Args:
            depth (int): The maximum depth of module hierarchy to display (default to 2).
        """
        print("Pre-Forward Execution Order: ")
        for mod_fqn in self.mod_fw_pre_order:
            mod_depth = mod_fqn.count(".") + 1
            if mod_depth > depth:
                continue
            print(mod_fqn)
        print("Pre-Backward Execution Order: ")
        for mod_fqn in self.mod_bw_pre_order:
            mod_depth = mod_fqn.count(".") + 1
            if mod_depth > depth:
                continue
            print(mod_fqn)
        for mod_fqn, runtimes in self.mod_runtimes.items():
            mod_depth = mod_fqn.count(".") + 1
            if mod_depth > depth:
                continue
            print(
                f"{mod_fqn} fw: {runtimes.get('fw', 0.0):.3f}ms bw: {runtimes.get('bw', 0.0):.3f}ms"
            )

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):  # type: ignore[no-untyped-def]
        # TODO: @sanketpurandare: Flatten tensors by desugaring the tensor subclasses
        # TODO: @sanketpurandare: Add logic for incorporating communication time
        res, op_time = self._estimate(func, args, kwargs)
        for par in self._mod_tracker.parents:
            if self._mod_tracker.is_bw:
                self.mod_runtimes[par]["bw"] += op_time
            else:
                self.mod_runtimes[par]["fw"] += op_time
        self.total_runtime += op_time
        return res

    def __call__(self, estimate_mode_type: str) -> Self:
        """
        Sets the estimate mode type.

        Currently supported modes:
            - "operator-level-benchmark": Estimates runtime using operator benchmarking.
            - "operator-level-cost-model": Estimates runtime using roofline cost model.
            - "operator-level-learned-cost-model": Estimates runtime using learned roofline cost model.
            - "operator-level-learned-model": Estimates runtime using learned model (random forest).

        Args:
            estimate_mode_type (str): The type of estimate mode to use.

        Returns:
            RuntimeEstimator: The runtime estimator instance.

        Raises:
            NotImplementedError: If the estimate mode type is not supported.
        """
        if estimate_mode_type == "operator-level-benchmark":
            self._estimate = RuntimeEstimator._benchmark_estimate
        elif estimate_mode_type == "operator-level-cost-model":
            self._estimate = RuntimeEstimator._roofline_estimate
        # elif estimate_mode_type == "operator-level-learned-cost-model":
            # self._estimate = RuntimeEstimator._learned_roofline_estimate
        elif estimate_mode_type == "operator-level-learned-model":
            self._estimate = RuntimeEstimator._learned_estimate
        else:
            raise NotImplementedError(
                f"estimate_mode_type {estimate_mode_type} not supported"
            )
        self._estimate_mode_type = estimate_mode_type
        return self

    def __enter__(self) -> Self:
        fake_mode = active_fake_mode()
        assert isinstance(
            fake_mode, FakeTensorMode
        ), "No FakeTensorMode found, designed to used under FakeTensorMode"
        RuntimeEstimator.fake_mode = fake_mode
        self.total_runtime = 0.0
        self.mod_runtimes = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.mod_fw_pre_order.clear()
        self.mod_bw_pre_order.clear()
        self.mod_fw_post_order.clear()
        self.mod_bw_post_order.clear()
        self._mod_tracker.register_user_hooks(
            pre_fw_hook=lambda mod, inp: self.mod_fw_pre_order.append(
                self._mod_tracker.get_known_fqn(mod)
            ),
            pre_bw_hook=lambda mod, g_out: self.mod_bw_pre_order.append(
                self._mod_tracker.get_known_fqn(mod)
            ),
            post_fw_hook=lambda mod, inp, out: self.mod_fw_post_order.append(
                self._mod_tracker.get_known_fqn(mod)
            ),
            post_bw_hook=lambda mod, g_inp: self.mod_bw_post_order.append(
                self._mod_tracker.get_known_fqn(mod)
            ),
        )
        self._mod_tracker.__enter__()
        super().__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        print(
            f"Estimated ({self._estimate_mode_type})"
            f"total_time: {self.total_runtime:.3f} ms"
        )
        if len(self._no_fallback_kernel) > 0:
            print("no_fallback_kernel: ", list(self._no_fallback_kernel))
        super().__exit__(*args)
        self._mod_tracker.clear_user_hooks()
        self._mod_tracker.__exit__()
