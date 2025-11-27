import pytest
import torch
from torch.func import vjp, vmap, jvp


@pytest.mark.parametrize("device", ["cpu", "xpu"])
def test_vmap_jvp_vjp(device):
    primal_in = torch.tensor(12.34, device=device, requires_grad=True)
    cotangent_in = torch.tensor([1.0, 3.0, 2.0, 4.5], device=device)
    
    def push_vjp(primal_in, cotangent_in):
        _, vjp_fn = vjp(torch.nn.functional.logsigmoid, primal_in)
        (grad,) = vjp_fn(cotangent_in)
        return grad
    
    def jvp_of_vjp(primal_in, cotangent_in, primal_tangent_in, cotangent_tangent_in):
        return jvp(
            push_vjp,
            (primal_in, cotangent_in),
            (primal_tangent_in, cotangent_tangent_in),
        )

    # Compare VMap (throwing error) vs looping individual cotangent_ins
    vmap_results = vmap(jvp_of_vjp, in_dims=(None, 0, None, None))(
        primal_in,
        cotangent_in,
        torch.tensor(1.0, device=device),
        torch.tensor(1.0, device=device)
    )

    batch_size = cotangent_in.shape[0]
    loop_primal_outs = torch.empty(batch_size, device=device)
    loop_tangent_outs = torch.empty(batch_size, device=device)
    
    for i, each_cotangent_in in enumerate(cotangent_in):
        primal_out, tangent_out = jvp_of_vjp(
            primal_in,
            each_cotangent_in,
            torch.tensor(1.0, device=device),
            torch.tensor(1.0, device=device)
        )
        loop_primal_outs[i] = primal_out
        loop_tangent_outs[i] = tangent_out

    torch.testing.assert_close(vmap_results[0], loop_primal_outs)
    torch.testing.assert_close(vmap_results[1], loop_tangent_outs)
