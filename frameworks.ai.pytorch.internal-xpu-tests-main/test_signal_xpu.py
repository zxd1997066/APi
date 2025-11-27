import torch
import pytest
import json


def verify_xpu_execution(check_traces, operator, *args, **kwargs):
    prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.XPU], record_shapes=True, profile_memory=True, with_stack=True)

    if check_traces:
        with prof:
            result = operator(*args, **kwargs, device="xpu")

        test_trace = "test_trace.json"
        prof.export_chrome_trace(test_trace)

        is_xpu_runtime = False
        is_xpu_kernel = False

        with open(test_trace) as f:
            trace = json.load(f)
            for t in trace["traceEvents"]:
                if not is_xpu_runtime and "cat" in t and t["cat"] == "xpu_runtime":
                    is_xpu_runtime = True
                if not is_xpu_kernel and "cat" in t and t["cat"] == "kernel" and "xpu" in t["name"]:
                    is_xpu_kernel = True
                if is_xpu_runtime and is_xpu_kernel:
                    break

        assert is_xpu_runtime, "XPU runtime category not found in trace"
        assert is_xpu_kernel, "XPU kernel not found in trace"
    else:
        result = operator(*args, **kwargs, device="xpu")

    result_ref = operator(*args, **kwargs, device="cpu")
    assert torch.allclose(result.cpu(), result_ref)
    assert result.dtype == kwargs["dtype"]
    assert result.device.type == "xpu"


@pytest.mark.parametrize("M", [0, 1, 5, 13])
@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("op", ["bartlett", "blackman", "cosine", "hamming", "hann", "nuttall"])
def test_windows(M, sym, dtype, op):
    operator = getattr(torch.signal.windows, op)

    check_traces = M != 0
    verify_xpu_execution(check_traces, operator, M, sym=sym, dtype=dtype)


@pytest.mark.parametrize("M", [0, 1, 5, 13])
@pytest.mark.parametrize("center", [None, 1, 7])
@pytest.mark.parametrize("tau", [1.0, 42.0, 100.0])
@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("op", ["exponential"])
def test_exponential(M, center, tau, sym, dtype, op):
    if sym and center:
        pytest.skip("Incompatible parameters")

    operator = getattr(torch.signal.windows, op)

    check_traces = M != 0
    verify_xpu_execution(check_traces, operator, M, center=center, tau=tau, sym=sym, dtype=dtype)


@pytest.mark.parametrize("M", [0, 1, 5, 13])
@pytest.mark.parametrize("std", [1.0, 3.14])
@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("op", ["gaussian"])
def test_gaussian(M, std, sym, dtype, op):
    operator = getattr(torch.signal.windows, op)

    check_traces = M != 0
    verify_xpu_execution(check_traces, operator, M, std=std, sym=sym, dtype=dtype)


@pytest.mark.parametrize("M", [0, 1, 5, 13])
@pytest.mark.parametrize("a", [[0.46, 0.23, 0.31], [0.5, 0.7]])
@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("op", ["general_cosine"])
def test_general_cosine(M, a, sym, dtype, op):
    operator = getattr(torch.signal.windows, op)

    check_traces = M != 0
    verify_xpu_execution(check_traces, operator, M, a=a, sym=sym, dtype=dtype)


@pytest.mark.parametrize("M", [0, 1, 5, 13])
@pytest.mark.parametrize("alpha", [0.46, 0.23, 0.54])
@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("op", ["general_hamming"])
def test_general_hamming(M, alpha, sym, dtype, op):
    operator = getattr(torch.signal.windows, op)

    check_traces = M != 0
    verify_xpu_execution(check_traces, operator, M, alpha=alpha, sym=sym, dtype=dtype)


@pytest.mark.parametrize("M", [0, 1, 5, 13])
@pytest.mark.parametrize("beta", [1.0, 12.0, 42.0])
@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("op", ["kaiser"])
def test_kaiser(M, beta, sym, dtype, op):
    operator = getattr(torch.signal.windows, op)

    check_traces = M != 0
    verify_xpu_execution(check_traces, operator, M, beta=beta, sym=sym, dtype=dtype)


