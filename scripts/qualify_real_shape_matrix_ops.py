"""Qualify real policy matrix/convolution/reduction/SDPA shapes over 100 seeds.

The JSON configuration file supplies source-observed dimensions and layouts;
this script records actual operand strides, singleton/batch invariance,
singleton fidelity, selected FP64 references, and raw native/explicit-reference/
invariant times.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_invariant_ops import (
    bmm_persistent, conv1d_batch_invariant, conv2d_batch_invariant,
    conv3d_batch_invariant, conv_transpose2d_batch_invariant,
    conv_transpose3d_batch_invariant,
    matmul_persistent, mean_dim, softmax as softmax_batch_invariant,
    scaled_dot_product_attention_batch_invariant,
    set_batch_invariant_mode,
)
from batch_invariant_ops.batch_invariant_ops import addmm_batch_invariant
from scripts.run_model_replay_invariance import metrics


def make_operands(config: dict, batch_size: int, seed: int):
    dtype_name = config.get("dtype", "float32").removeprefix("torch.")
    if dtype_name not in ("float32", "bfloat16", "float16"):
        raise ValueError(f"unsupported operator dtype: {dtype_name}")
    dtype = getattr(torch, dtype_name)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    layout = config["layout"]
    if config["operation"] == "matmul":
        def operand(which):
            shape = list(config[f"source_{which}_shape"])
            shape[0] = batch_size
            stride = tuple(config[f"source_{which}_stride_b1" if batch_size == 1 else f"source_{which}_stride_b8"])
            if any(s < 0 for s in stride):raise ValueError("negative source stride unsupported")
            # Fill physical storage once. Direct writes through a zero-stride
            # expanded view would overlap and would not model broadcasting.
            storage_size = 1 + sum((n-1)*s for n,s in zip(shape,stride))
            storage = torch.randn(storage_size,device="cuda",dtype=dtype,generator=generator)
            return storage.as_strided(shape,stride)
        return operand("left"),operand("right"),None
    if config["operation"] in ("mean", "softmax"):
        left = torch.randn(
            batch_size * config["m_per_request"], *config["input_per_unit"],
            device="cuda", dtype=dtype, generator=generator,
        )
        return left, None, None
    if config["operation"] == "sdpa":
        query_heads = config.get("query_heads", config.get("heads"))
        key_heads = config.get("key_heads", config.get("heads"))
        value_heads = config.get("value_heads", config.get("heads"))
        query_length, key_length = config["query_length"], config["key_length"]
        head_dim, value_dim = config["head_dim"], config["value_dim"]
        outer = batch_size * config["m_per_request"]
        query_stride = tuple(config.get("source_q_stride", config.get("source_qkv_stride")))
        key_stride = tuple(config.get("source_k_stride", config.get("source_qkv_stride")))
        value_stride = tuple(config.get("source_v_stride", config.get("source_qkv_stride")))
        def source_strided(shape, stride):
            operand = torch.empty_strided(shape, stride, device="cuda", dtype=dtype)
            operand.normal_(generator=generator)
            return operand
        query = source_strided((outer, query_heads, query_length, head_dim), query_stride)
        key = source_strided((outer, key_heads, key_length, head_dim), key_stride)
        value = source_strided((outer, value_heads, key_length, value_dim), value_stride)
        return query, key, value
    if config["operation"] in (
        "conv1d", "conv2d", "conv3d", "conv_transpose2d", "conv_transpose3d",
    ):
        input_shape = config.get("input_per_unit", config.get("input_per_image"))
        left = torch.randn(
            batch_size * config["m_per_request"], *input_shape,
            device="cuda", dtype=dtype, generator=generator,
        )
        if layout == "channels_last":
            left = left.contiguous(memory_format=torch.channels_last)
        elif layout == "channels_last_3d":
            left = left.contiguous(memory_format=torch.channels_last_3d)
        right = torch.randn(
            *config["weight_shape"], device="cuda", dtype=dtype,
            generator=generator,
        )
        output_channels = (
            config["weight_shape"][1] * config.get("groups", 1)
            if config["operation"] in ("conv_transpose2d", "conv_transpose3d") else
            config["weight_shape"][0]
        )
        bias = (
            torch.randn(output_channels, device="cuda", dtype=dtype, generator=generator)
            if config.get("has_bias", True) else None
        )
        return left, right, bias
    if config["operation"] == "bmm":
        heads = batch_size * config["m_per_request"]
        m, k, n = config["m"], config["k"], config["n"]
        if layout == "left_strided":
            left = torch.randn(heads, m, k * 2, device="cuda", dtype=dtype, generator=generator)[..., ::2]
        else:
            left = torch.randn(heads, m, k, device="cuda", dtype=dtype, generator=generator)
        if layout == "right_transposed":
            right = torch.randn(heads, n, k, device="cuda", dtype=dtype, generator=generator).transpose(1, 2)
        else:
            right = torch.randn(heads, k, n, device="cuda", dtype=dtype, generator=generator)
        return left, right, None
    per_request_m = config["m_per_request"]
    k = config["k"]
    n = config["n"]
    if layout == "left_strided":
        left_storage = torch.randn(
            batch_size * per_request_m, k * 2,
            device="cuda", dtype=dtype, generator=generator,
        )
        left = left_storage[:, ::2]
    else:
        left = torch.randn(
            batch_size * per_request_m, k,
            device="cuda", dtype=dtype, generator=generator,
        )
    if layout == "right_transposed":
        right_storage = torch.randn(n, k, device="cuda", dtype=dtype, generator=generator)
        right = right_storage.t()
    else:
        right = torch.randn(k, n, device="cuda", dtype=dtype, generator=generator)
    bias = (
        torch.randn(n, device="cuda", dtype=dtype, generator=generator)
        if config["operation"] == "addmm" else None
    )
    return left, right, bias


def call(operation: str, left, right, bias, *, invariant: bool, config: dict):
    if operation == "matmul":
        # Exercise the same ATen composite and its mm/bmm dispatch as the
        # complete policy, including source broadcast/reshape decisions.
        with torch.inference_mode(), set_batch_invariant_mode(invariant):
            return torch.matmul(left,right)
    if operation == "mean":
        return (
            mean_dim(left, config["dim"], keepdim=config["keepdim"])
            if invariant else
            torch.mean(left, dim=config["dim"], keepdim=config["keepdim"])
        )
    if operation == "softmax":
        output_dtype = getattr(torch, config["output_dtype"].removeprefix("torch.")) if config.get("output_dtype") else None
        return (
            softmax_batch_invariant(left, dim=config["dim"], dtype=output_dtype)
            if invariant else torch.softmax(left, dim=config["dim"], dtype=output_dtype)
        )
    if operation == "sdpa":
        function = scaled_dot_product_attention_batch_invariant if invariant else F.scaled_dot_product_attention
        return function(
            left, right, bias, dropout_p=0.0, is_causal=False,
            scale=config["scale"], enable_gqa=config.get("enable_gqa", False),
        )
    if operation == "mm":
        return matmul_persistent(left, right) if invariant else torch.mm(left, right)
    if operation == "addmm":
        return addmm_batch_invariant(bias, left, right) if invariant else torch.addmm(bias, left, right)
    if operation in (
        "conv1d", "conv2d", "conv3d", "conv_transpose2d", "conv_transpose3d",
    ):
        stride = tuple(config["stride"])
        padding = tuple(config["padding"])
        spatial_dimensions = (
            2 if operation == "conv_transpose2d" else
            3 if operation == "conv_transpose3d" else int(operation[-2])
        )
        dilation = tuple(config.get("dilation", (1,) * spatial_dimensions))
        output_padding = tuple(config.get("output_padding", (0,) * spatial_dimensions))
        groups = config.get("groups", 1)
        invariant_function = {
            "conv1d": conv1d_batch_invariant,
            "conv2d": conv2d_batch_invariant,
            "conv3d": conv3d_batch_invariant,
            "conv_transpose2d": conv_transpose2d_batch_invariant,
            "conv_transpose3d": conv_transpose3d_batch_invariant,
        }[operation]
        if operation in ("conv_transpose2d", "conv_transpose3d"):
            native_transpose = (
                F.conv_transpose2d if operation == "conv_transpose2d" else F.conv_transpose3d
            )
            return (invariant_function(
                left, right, bias, stride, padding, dilation, output_padding, groups,
            ) if invariant else native_transpose(
                left, right, bias, stride, padding, output_padding, groups, dilation,
            ))
        native_function = {"conv1d": F.conv1d, "conv2d": F.conv2d, "conv3d": F.conv3d}[operation]
        return (invariant_function(
            left, right, bias, stride, padding, dilation, False, output_padding, groups,
        ) if invariant else native_function(left, right, bias, stride, padding, dilation, groups))
    if operation == "bmm":
        return bmm_persistent(left, right) if invariant else torch.bmm(left, right)
    raise ValueError(operation)


def explicit_per_matrix_reference(operation: str, left, right, bias, config: dict):
    """Execute each independent request/image/head at its singleton shape."""
    if operation == "matmul":
        with torch.inference_mode():
            return torch.cat([torch.matmul(left[i:i+1],right[i:i+1]) for i in range(left.shape[0])],dim=0)
    if operation == "mean":
        rows = config["m_per_request"]
        return torch.cat([
            torch.mean(
                left[index : index + rows],
                dim=config["dim"], keepdim=config["keepdim"],
            )
            for index in range(0, left.shape[0], rows)
        ], dim=0)
    if operation == "softmax":
        rows = config["m_per_request"]
        output_dtype = getattr(torch, config["output_dtype"].removeprefix("torch.")) if config.get("output_dtype") else None
        return torch.cat([
            torch.softmax(left[index : index + rows], dim=config["dim"], dtype=output_dtype)
            for index in range(0, left.shape[0], rows)
        ], dim=0)
    if operation == "sdpa":
        if config.get("enable_gqa", False):
            repeats = left.shape[1] // right.shape[1]
            right = right.repeat_interleave(repeats, dim=1)
            bias = bias.repeat_interleave(repeats, dim=1)
        return torch.cat([
            torch.cat([
                F.scaled_dot_product_attention(
                    left[batch : batch + 1, head : head + 1],
                    right[batch : batch + 1, head : head + 1],
                    bias[batch : batch + 1, head : head + 1],
                    dropout_p=0.0, is_causal=False, scale=config["scale"],
                )
                for head in range(left.shape[1])
            ], dim=1)
            for batch in range(left.shape[0])
        ], dim=0)
    if operation in ("mm", "addmm"):
        rows = config["m_per_request"]
        outputs = []
        for start in range(0, left.shape[0], rows):
            operand = left[start : start + rows]
            outputs.append(
                torch.mm(operand, right) if operation == "mm"
                else torch.addmm(bias, operand, right)
            )
        return torch.cat(outputs, dim=0)
    if operation == "bmm":
        return torch.stack(
            [torch.mm(left[index], right[index]) for index in range(left.shape[0])],
            dim=0,
        )
    if operation in (
        "conv1d", "conv2d", "conv3d", "conv_transpose2d", "conv_transpose3d",
    ):
        spatial_dimensions = (
            2 if operation == "conv_transpose2d" else
            3 if operation == "conv_transpose3d" else int(operation[-2])
        )
        function = {
            "conv1d": F.conv1d, "conv2d": F.conv2d, "conv3d": F.conv3d,
            "conv_transpose3d": F.conv_transpose3d,
            "conv_transpose2d": F.conv_transpose2d,
        }[operation]
        rows = config["m_per_request"]
        def per_request(value):
            if operation in ("conv_transpose2d", "conv_transpose3d"):
                return function(
                    value, right, bias, tuple(config["stride"]), tuple(config["padding"]),
                    tuple(config.get("output_padding", (0,) * spatial_dimensions)),
                    config.get("groups", 1),
                    tuple(config.get("dilation", (1,) * spatial_dimensions)),
                )
            return function(
                value, right, bias, tuple(config["stride"]), tuple(config["padding"]),
                tuple(config.get("dilation", (1,) * spatial_dimensions)),
                config.get("groups", 1),
            )
        return torch.cat([
            per_request(left[index : index + rows])
            for index in range(0, left.shape[0], rows)
        ], dim=0)
    raise ValueError(operation)


def time_call(function) -> float:
    function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    function()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000


def profile_dispatch(operation: str, left, right, bias, config: dict, _attempt: int = 0) -> dict:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as trace:
        call(operation, left, right, bias, invariant=False, config=config)
        call(operation, left, right, bias, invariant=True, config=config)
        torch.cuda.synchronize()
    aten_events = sorted({event.key for event in trace.key_averages() if event.key.startswith("aten::")})
    expected = (
        f"aten::{operation}" if operation in ("conv_transpose2d", "conv_transpose3d") else
        "aten::convolution" if operation.startswith("conv") else
        "aten::scaled_dot_product_attention" if operation == "sdpa" else
        "aten::_softmax" if operation == "softmax" else
        f"aten::{operation}"
    )
    if expected not in aten_events:
        raise RuntimeError(f"expected dispatched {expected} was absent from profiler")
    cuda_kernels = sorted({
        event.name for event in trace.events()
        if getattr(getattr(event, "device_type", None), "name", "") == "CUDA"
    })
    if not cuda_kernels:
        # CUPTI can return an empty device trace despite a completed CUDA
        # execution. Retry capture, never substitute expected kernel names or
        # accept CPU-only profiling evidence. Bound retries and retain count.
        if _attempt < 2:
            return profile_dispatch(operation, left, right, bias, config, _attempt + 1)
        raise RuntimeError("profiler did not capture a native or invariant CUDA kernel")
    return {"aten_events": aten_events, "cuda_kernels": cuda_kernels, "capture_attempt": _attempt + 1}


def completed(path: Path, configuration_hash: str) -> set[tuple]:
    keys = set()
    if not path.is_file():
        return keys
    with path.open() as stream:
        for line in stream:
            item = json.loads(line)
            if item.get("configuration_sha256") != configuration_hash:
                raise ValueError("existing operator run uses another source-shape configuration")
            key = (item["operator_configuration"], item["seed"])
            if key in keys:
                raise ValueError(f"duplicate existing operator record {key}")
            keys.add(key)
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(1, 2, 4, 8, 16, 31, 32, 33, 64))
    parser.add_argument("--configuration-ids", nargs="+", help="run only these IDs while retaining the full manifest hash")
    parser.add_argument("--arrangement-keys", nargs="+", help="run selected ID|Bsize|position keys in isolated profiler processes")
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.seeds != 100:
        raise ValueError("full operator qualification requires exactly 100 seeds/configuration")
    if set(args.batch_sizes) != {1, 2, 4, 8, 16, 31, 32, 33, 64}:
        raise ValueError("full operator batch sweep requires B=1,2,4,8,16,31,32,33,64")
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    configuration_hash = hashlib.sha256(args.config.read_bytes()).hexdigest()
    manifest = json.loads(args.config.read_text())
    precision = manifest.get("adapter_execution_configuration", {}).get("tf32")
    if precision is not None:
        if not isinstance(precision, bool):raise ValueError("TF32 configuration must be boolean")
        torch.backends.cuda.matmul.allow_tf32 = precision
        torch.backends.cudnn.allow_tf32 = precision
        torch.backends.cudnn.benchmark = False
    runtime_identity = {
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "gpu_uuid": str(getattr(torch.cuda.get_device_properties(torch.cuda.current_device()), "uuid", "unavailable")),
        "qualifier_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    configurations = manifest["configurations"]
    if not configurations:
        raise ValueError("empty operator configuration file")
    if args.configuration_ids:
        known_ids = {config["id"] for config in configurations}
        unknown_ids = set(args.configuration_ids) - known_ids
        if unknown_ids:
            raise ValueError(f"unknown operator configuration IDs: {sorted(unknown_ids)}")
        configurations = [config for config in configurations if config["id"] in args.configuration_ids]
    done = completed(args.output, configuration_hash)
    if args.arrangement_keys:
        valid = {
            f"{config['id']}|B{size}|{position}"
            for config in configurations
            for size in config.get("batch_sizes", args.batch_sizes)
            for position in (("first",) if size == 1 else ("first", "last"))
        }
        if set(args.arrangement_keys) - valid:
            raise ValueError("unknown operator arrangement key")
    for config in configurations:
        if config["operation"] not in ("mm", "addmm", "bmm", "matmul", "conv1d", "conv2d", "conv3d", "conv_transpose2d", "conv_transpose3d", "mean", "softmax", "sdpa"):
            raise ValueError(config["operation"])
        if config["layout"] not in ("contiguous", "left_strided", "right_transposed", "channels_last", "channels_last_3d", "qkv_permuted", "qkv_source_strided", "source_strided_matmul"):
            raise ValueError(config["layout"])
        if (config["operation"] == "sdpa") != (config["layout"] in ("qkv_permuted", "qkv_source_strided")):
            raise ValueError("SDPA requires an observed Q/K/V stride layout")
        if config["operation"] not in ("conv2d", "conv_transpose2d") and config["layout"] == "channels_last":
            raise ValueError("channels_last applies only to Conv2d/ConvTranspose2d")
        if config["operation"] not in ("conv3d", "conv_transpose3d") and config["layout"] == "channels_last_3d":
            raise ValueError("channels_last_3d applies only to Conv3d/ConvTranspose3d")
        if config["operation"] in ("mean", "softmax") and (
            config.get("layout") != "contiguous"
            or not isinstance(config.get("dim"), int)
            or (config["operation"] == "mean" and not isinstance(config.get("keepdim"), bool))
            or config["dim"] in (0, -len(config.get("input_per_unit", [])) - 1)
        ):
            raise ValueError(
                f"{config['operation']} qualification must preserve the leading request dimension"
            )
        configured_sizes = tuple(config.get("batch_sizes", args.batch_sizes))
        if not set(configured_sizes).issubset(args.batch_sizes):
            raise ValueError(f"{config['id']} requires batch sizes absent from --batch-sizes")
        for batch_size in configured_sizes:
            positions = ("first",) if batch_size == 1 else ("first", "last")
            for target_position in positions:
                configuration_key = f"{config['id']}|B{batch_size}|{target_position}"
                if args.arrangement_keys and configuration_key not in args.arrangement_keys:
                    continue
                for seed_index in range(args.seeds):
                    key = (configuration_key, seed_index)
                    if key in done:
                        continue
                    seed = 1_000_000 + seed_index
                    left, right, bias = make_operands(config, batch_size, seed)
                    start_row = 0 if target_position == "first" else (batch_size - 1) * config["m_per_request"]
                    target_left = left[start_row : start_row + config["m_per_request"]]
                    operation = config["operation"]
                    target_right = right[start_row : start_row + config["m_per_request"]] if operation in ("bmm", "sdpa", "matmul") else right
                    target_bias = bias[start_row : start_row + config["m_per_request"]] if operation == "sdpa" else bias
                    native_singleton = call(operation, target_left, target_right, target_bias, invariant=False, config=config)
                    native_batched = call(operation, left, right, bias, invariant=False, config=config)
                    invariant_singleton = call(operation, target_left, target_right, target_bias, invariant=True, config=config)
                    invariant_batched = call(operation, left, right, bias, invariant=True, config=config)
                    reference_singleton = explicit_per_matrix_reference(operation, target_left, target_right, target_bias, config)
                    reference_batched = explicit_per_matrix_reference(operation, left, right, bias, config)
                    torch.cuda.synchronize()
                    target_slice = slice(start_row, start_row + config["m_per_request"])
                    native_comparison = metrics(native_singleton, native_batched[target_slice])
                    invariant_comparison = metrics(invariant_singleton, invariant_batched[target_slice])
                    fidelity = metrics(native_singleton, invariant_singleton)
                    reference_comparison = metrics(reference_singleton, reference_batched[target_slice])
                    reference_fidelity = metrics(native_singleton, reference_singleton)
                    reference = None
                    if seed_index in (0, 1, 2):
                        double_output = call(
                            operation, target_left.double(),
                            target_right.double() if target_right is not None else None,
                            target_bias.double() if target_bias is not None else None,
                            invariant=False, config={**config, "output_dtype": "float64"} if operation == "softmax" else config,
                        )
                        reference = {
                            "native": metrics(double_output, native_singleton.double()),
                            "invariant": metrics(double_output, invariant_singleton.double()),
                        }
                    dispatch_profile = (
                        profile_dispatch(operation, left, right, bias, config)
                        if seed_index == 0 else None
                    )
                    record = {
                        **runtime_identity,
                        "schema_version": 1,
                        "configuration_sha256": configuration_hash,
                        "operator_configuration": configuration_key,
                        "seed": seed_index,
                        "generator_seed": seed,
                        "source_operation": config["source_operation"],
                        "dispatched_aten_operation": (
                            f"aten::{operation}"
                            if operation in ("conv_transpose2d", "conv_transpose3d") else
                            "aten::convolution" if operation.startswith("conv") else
                            "aten::scaled_dot_product_attention" if operation == "sdpa" else
                            "aten::_softmax" if operation == "softmax" else
                            f"aten::{operation}"
                        ),
                        "left_shape": list(left.shape),
                        "right_shape": list(right.shape) if right is not None else None,
                        "left_stride": list(left.stride()),
                        "right_stride": list(right.stride()) if right is not None else None,
                        "value_shape": list(bias.shape) if operation == "sdpa" else None,
                        "value_stride": list(bias.stride()) if operation == "sdpa" else None,
                        "dtype": str(left.dtype),
                        "layout": config["layout"],
                        "batch_size": batch_size,
                        "target_position": target_position,
                        "native_batch_invariance": native_comparison,
                        "invariant_batch_invariance": invariant_comparison,
                        "singleton_fidelity": fidelity,
                        "reference_batch_invariance": reference_comparison,
                        "reference_singleton_fidelity": reference_fidelity,
                        "fp64_reference": reference,
                        "dispatch_profile": dispatch_profile,
                        "native_latency_seconds": time_call(lambda: call(operation, left, right, bias, invariant=False, config=config)),
                        "invariant_latency_seconds": time_call(lambda: call(operation, left, right, bias, invariant=True, config=config)),
                        "reference_latency_seconds": time_call(lambda: explicit_per_matrix_reference(operation, left, right, bias, config)),
                    }
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    with args.output.open("a") as stream:
                        stream.write(json.dumps(record, sort_keys=True) + "\n")
                    done.add(key)
                print(f"qualified {configuration_key}: 100 seeds", flush=True)
    print(args.output)


if __name__ == "__main__":
    main()
