import torch
import pytest

class TestSpecialXPU:
    SUPPORTED_DTYPES = [torch.float32, torch.bfloat16, torch.half, torch.double, torch.uint8, torch.int8, torch.bool]

    def _generate_input(self, shape, input_type, dtype):
        if dtype in [torch.float32, torch.double, torch.bfloat16, torch.half]:
            if input_type == "normal":
                return torch.randn(shape, dtype=dtype)
            elif input_type == "uniform":
                return torch.rand(shape, dtype=dtype)
            elif input_type == "positive":
                return torch.rand(shape, dtype=dtype) + 1.0
            elif input_type == "positive_small":
                return torch.rand(shape, dtype=dtype) + 0.1
            elif input_type == "range_neg1_1":
                return torch.rand(shape, dtype=dtype) * 2 - 1
            else:
                raise ValueError(f"Unknown input_type: {input_type}")
        elif dtype in [torch.uint8, torch.int8, torch.bool]:
            if input_type == "normal":
                if dtype == torch.uint8:
                    return torch.randint(0, 256, shape, dtype=dtype)
                elif dtype == torch.int8:
                    return torch.randint(-128, 128, shape, dtype=dtype)
                else:
                    return torch.randn(shape) > 0
            elif input_type == "uniform":
                if dtype == torch.uint8:
                    return torch.randint(0, 256, shape, dtype=dtype)
                elif dtype == torch.int8:
                    return torch.randint(-128, 128, shape, dtype=dtype)
                else:
                    return torch.rand(shape) > 0.5
            elif input_type == "positive":
                if dtype == torch.uint8:
                    return torch.randint(1, 256, shape, dtype=dtype)
                elif dtype == torch.int8:
                    return torch.randint(1, 128, shape, dtype=dtype)
                else:
                    return torch.ones(shape, dtype=dtype)
            elif input_type == "positive_small":
                if dtype == torch.uint8:
                    return torch.randint(1, 11, shape, dtype=dtype)
                elif dtype == torch.int8:
                    return torch.randint(1, 11, shape, dtype=dtype)
                else:
                    return torch.ones(shape, dtype=dtype)
            elif input_type == "range_neg1_1":
                if dtype == torch.uint8:
                    return torch.randint(0, 2, shape, dtype=dtype)  # 0 or 1
                elif dtype == torch.int8:
                    return torch.randint(-1, 2, shape, dtype=dtype)  # -1, 0, or 1
                else:
                    return torch.rand(shape) > 0.5
            else:
                raise ValueError(f"Unknown input_type: {input_type}")
        
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def _get_dtype_tolerance(self, dtype):
        if dtype in [torch.float32, torch.double]:
            return 1e-5, 1e-5
        elif dtype is torch.bfloat16:
            return 16e-3, 1e-5
        elif dtype is torch.half:
            return 1.2e-03, 1e-03
        elif dtype in [torch.uint8, torch.int8, torch.bool]:
            return 1e-3, 1e-2
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def _test_special_op(self, op_name, *args, **kwargs):
        op = getattr(torch.special, op_name)
        
        try:
            cpu_args = [arg.cpu() if isinstance(arg, torch.Tensor) else arg for arg in args]
            cpu_result = op(*cpu_args, **kwargs)
        except (ValueError, RuntimeError) as e:
            pytest.skip(f"CPU version of '{op_name}' failed with: {e}")


        xpu_args = [arg.xpu() if isinstance(arg, torch.Tensor) else arg for arg in args]
        xpu_result = op(*xpu_args, **kwargs)

        print("cpu_result",cpu_result)
        print("xpu_result",xpu_result)
        print("diff", cpu_result - xpu_result.cpu())

        rtol, atol = self._get_dtype_tolerance(xpu_result.dtype)
        torch.testing.assert_close(xpu_result.cpu(), cpu_result, rtol=rtol, atol=atol, equal_nan=True)

        # Check that both CPU and XPU results have NaN on same indices
        cpu_nan_mask = torch.isnan(cpu_result)
        xpu_nan_mask = torch.isnan(xpu_result.cpu())
        if not torch.equal(cpu_nan_mask, xpu_nan_mask):
            raise AssertionError(f"NaN positions don't match. CPU: {cpu_nan_mask}, XPU: {xpu_nan_mask}")        

    @pytest.mark.parametrize("op_name", [
        "airy_ai",
        "bessel_j0",
        "bessel_j1",
        "bessel_y0",
        "bessel_y1",
        "digamma",
        "entr",
        "erf",
        "erfc",
        "erfcx",
        "erfinv",
        "exp2",
        "expit",
        "expm1",
        "gammaln",
        "i0",
        "i0e",
        "i1",
        "i1e",
        "log1p",
        "log_ndtr",
        "modified_bessel_i0",
        "modified_bessel_i1",
        "modified_bessel_k0",
        "modified_bessel_k1",
        "ndtr",
        "ndtri",
        "psi",
        "round",
        "scaled_modified_bessel_k0",
        "scaled_modified_bessel_k1",
        "sinc",
        "spherical_bessel_j0",
    ])
    @pytest.mark.parametrize("shape", [(100,), (1,), (32, 12, 3)])
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    @pytest.mark.parametrize("input_type", [
        "normal", "uniform", "positive", "positive_small", "range_neg1_1"
    ])
    def test_special_unary_ops(self, op_name, shape, input_type, dtype):
        x = self._generate_input(shape, input_type, dtype)
        self._test_special_op(op_name, x)


    @pytest.mark.parametrize("op_name", [
        "chebyshev_polynomial_t",
        "chebyshev_polynomial_u",
        "chebyshev_polynomial_v",
        "chebyshev_polynomial_w",
        "hermite_polynomial_h",
        "hermite_polynomial_he",
        "laguerre_polynomial_l",
        "legendre_polynomial_p",
        "shifted_chebyshev_polynomial_t",
        "shifted_chebyshev_polynomial_u",
        "shifted_chebyshev_polynomial_v",
        "shifted_chebyshev_polynomial_w",
    ])
    @pytest.mark.parametrize("shape", [(100,), (1,), (32, 12, 3)])
    @pytest.mark.parametrize("input_type", [
        "normal", "uniform", "positive", "positive_small", "range_neg1_1"
    ])
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 10, 50, 500])
    def test_special_polynomial_ops(self, op_name, shape, input_type, dtype, n):
        x = self._generate_input(shape, input_type, dtype)
        self._test_special_op(op_name, x, n)


    @pytest.mark.parametrize("shape", [(100,), (1,), (32, 12, 3)])
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    @pytest.mark.parametrize("input_type", ["positive", "positive_small"])
    def test_special_gammainc_ops(self, shape, dtype, input_type):
        if (dtype in [torch.half, torch.bfloat16]) :
            pytest.skip("igamma_cuda not implemented for 'BFloat16'/'Half'")
        
        a = self._generate_input(shape, input_type, dtype)
        x = self._generate_input(shape, input_type, dtype)
        rtol, atol = self._get_dtype_tolerance(dtype)

        try:
            cpu_gammainc = torch.special.gammainc(a.cpu(), x.cpu())
        except (ValueError, RuntimeError) as e:
            pytest.skip(f"CPU version of 'gammainc' failed with: {e}")
        xpu_gammainc = torch.special.gammainc(a.xpu(), x.xpu())
        torch.testing.assert_close(xpu_gammainc.cpu(), cpu_gammainc, rtol=rtol, atol=atol)

        try:
            cpu_gammaincc = torch.special.gammaincc(a.cpu(), x.cpu())
        except (ValueError, RuntimeError) as e:
            pytest.skip(f"CPU version of 'gammaincc' failed with: {e}")        
        xpu_gammaincc = torch.special.gammaincc(a.xpu(), x.xpu())
        torch.testing.assert_close(xpu_gammaincc.cpu(), cpu_gammaincc, rtol=rtol, atol=atol)

        # test if gammainc + gammaincc == 1
        xpu_sum = xpu_gammainc + xpu_gammaincc
        expected_ones_xpu = torch.ones_like(xpu_sum)
        torch.testing.assert_close(xpu_sum.cpu(), expected_ones_xpu.cpu(), rtol=rtol, atol=atol)

    @pytest.mark.parametrize("shape", [(10, 5), (8, 4, 6), (2, 3, 4, 5)])
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    @pytest.mark.parametrize("input_type", ["normal", "uniform", "positive"])
    @pytest.mark.parametrize("dim", [0, 1, -1])
    def test_special_log_softmax(self, shape, dtype, input_type, dim):
        x = self._generate_input(shape, input_type, dtype)
        self._test_special_op("log_softmax", x, dim=dim)

    @pytest.mark.parametrize("shape", [(100,), (1,), (32, 12, 3)])
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    @pytest.mark.parametrize("input_type", ["uniform", "range_neg1_1"])
    @pytest.mark.parametrize("eps", [None, 1e-6, 1e-3, 0.01, 0.1])
    def test_special_logit(self, shape, dtype, input_type, eps):
        x = self._generate_input(shape, input_type, dtype)
        self._test_special_op("logit", x, eps=eps)


    @pytest.mark.parametrize("shape", [(8, 4, 6), (2, 3, 4, 5)])
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    @pytest.mark.parametrize("input_type", ["normal", "uniform", "positive", "positive_small", "range_neg1_1"])
    @pytest.mark.parametrize("dim", [0, 1, -1, (0, 1), (1, 2), (-2, -1)])
    @pytest.mark.parametrize("keepdim", [True, False])
    def test_special_logsumexp(self, shape, dtype, input_type, dim, keepdim):   
        x = self._generate_input(shape, input_type, dtype)
        self._test_special_op("logsumexp", x, dim, keepdim=keepdim)

    @pytest.mark.parametrize("shape", [(100,), (1,), (32, 12, 3)])
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    @pytest.mark.parametrize("p", [1, 2, 3, 4, 5])
    def test_special_multigammaln(self, shape, dtype, p):
        # All elements must be greater than (p-1)/2
        min_value = (p - 1) / 2 + 0.1  # Add small offset for safety
        x = self._generate_input(shape, "uniform", dtype) + min_value
        x = x.to(dtype)
        self._test_special_op("multigammaln", x, p)

    @pytest.mark.parametrize("shape", [(10, 5), (8, 4, 6), (2, 3, 4, 5)])
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    @pytest.mark.parametrize("input_type", ["normal", "uniform", "positive"])
    @pytest.mark.parametrize("n", [0, 1, 3, 5, 10])
    def test_special_polygamma(self, shape, dtype, input_type, n):
        x = self._generate_input(shape, input_type, dtype)
        self._test_special_op("polygamma", n, x)

    @pytest.mark.parametrize("shape", [(10, 5), (8, 4, 6), (2, 3, 4, 5)])
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    @pytest.mark.parametrize("expected_dtype", SUPPORTED_DTYPES)
    @pytest.mark.parametrize("input_type", ["normal", "uniform", "positive"])
    @pytest.mark.parametrize("dim", [0, 1, -1])
    def test_special_softmax(self, shape, dtype, input_type, dim,expected_dtype):
        x = self._generate_input(shape, input_type, dtype)
        self._test_special_op("softmax", x, dim=dim, dtype=expected_dtype)


    @pytest.mark.parametrize("shape", [(100,), (1,), (32, 12, 3)])
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    @pytest.mark.parametrize("input_type", ["normal", "uniform", "positive"])
    def test_special_xlog1py(self, shape, dtype, input_type):
        a = self._generate_input(shape, input_type, dtype)
        b = self._generate_input(shape, input_type, dtype)
        self._test_special_op("xlog1py", a, b)
        
    @pytest.mark.parametrize("shape", [(100,), (1,), (32, 12, 3)])
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    @pytest.mark.parametrize("input_type", ["normal", "uniform", "positive"])
    def test_special_xlogy(self, shape, dtype, input_type):
        a = self._generate_input(shape, input_type, dtype)
        b = self._generate_input(shape, input_type, dtype)
        self._test_special_op("xlogy", a, b)

    @pytest.mark.parametrize("shape", [(100,), (1,), (32, 12, 3)])
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    @pytest.mark.parametrize("input_type", ["normal", "uniform", "positive"])
    def test_special_zeta(self, shape, dtype, input_type):
        a = self._generate_input(shape, input_type, dtype)
        b = self._generate_input(shape, input_type, dtype)
        self._test_special_op("zeta", a, b)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["float32", "bfloat16"])
def test_special_logit(dtype):
    input_cpu = torch.tensor([0.5234], device="cpu", dtype=dtype)
    input_xpu = input_cpu.to("xpu")

    reference_cpu = torch.log(input_cpu/(1 - input_cpu))
    reference_xpu = torch.log(input_xpu/(1 - input_xpu))
    assert torch.allclose(reference_cpu, reference_xpu.cpu(), atol=1e-5, rtol=1e-5)

    logit_cpu = torch.special.logit(input_cpu)
    logit_xpu = torch.special.logit(input_xpu)
    assert torch.allclose(logit_cpu, logit_xpu.cpu(), atol=1e-5, rtol=1e-5)
