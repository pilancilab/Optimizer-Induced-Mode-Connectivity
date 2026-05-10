"""
Fix for Muon + HuggingFace GPT-2: handle fused c_attn (QKV) weight matrices.

Problem:
  HuggingFace GPT-2 fuses Q, K, V into a single Conv1D layer called `c_attn`
  with weight shape (n_embd, 3*n_embd). When Muon orthogonalizes this 2D matrix
  as a single unit, the resulting Q, K, V sub-matrices are NOT independently
  isotropic — they can appear low-rank.

  In Keller's code, qkvo_w has shape (4, hdim, dim) — a 3D tensor — so Muon's
  batched Newton-Schulz orthogonalizes each of Q, K, V, O independently.

Solution:
  Reshape c_attn.weight from (n_embd, 3*n_embd) to (3, n_embd, n_embd) before
  the Muon update, then reshape back. This makes Newton-Schulz treat Q, K, V
  as a batch of independent square matrices.

Usage:
  Replace your current `split_muon_params` + `MuonWithAuxAdam` setup with the
  helpers below.
"""

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Option A (recommended): Reshape c_attn in-place so Muon sees (3, d, d)
# ---------------------------------------------------------------------------

def reshape_cattn_for_muon(model):
    """
    Reshape every c_attn.weight from (n_embd, 3*n_embd) to (3, n_embd, n_embd)
    so that Muon's batched Newton-Schulz orthogonalizes Q, K, V independently.

    Call this BEFORE creating the optimizer.

    NOTE: This changes the parameter shape, so HuggingFace's forward() will
    break unless you also patch the forward method. See Option B if you prefer
    a non-invasive approach.
    """
    for name, module in model.named_modules():
        if hasattr(module, 'c_attn'):
            w = module.c_attn.weight  # shape: (n_embd, 3*n_embd)
            n_embd = w.shape[0]
            assert w.shape[1] == 3 * n_embd, (
                f"c_attn.weight shape {w.shape} doesn't match (n_embd, 3*n_embd)"
            )
            # Replace with a (3, n_embd, n_embd) parameter
            new_w = w.data.view(n_embd, 3, n_embd).permute(1, 0, 2).contiguous()
            module.c_attn.weight = torch.nn.Parameter(new_w)
            # You'll also need to patch the forward — see Option B instead.


# ---------------------------------------------------------------------------
# Option B (recommended, non-invasive): Custom optimizer that reshapes grads
# ---------------------------------------------------------------------------

def split_muon_params_fixed(model):
    """
    Split model parameters into Muon vs Adam groups, correctly handling
    HuggingFace GPT-2's fused c_attn weights.

    Returns: (muon_params, adam_params, cattn_param_names)

    The returned lists are (name, param) pairs so you can identify c_attn params.
    """
    muon_params = []
    adam_params = []
    cattn_params = []  # these need special handling

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        is_hidden_block = name.startswith("transformer.h.")
        is_matrix = (p.ndim >= 2)

        if is_hidden_block and is_matrix:
            if "c_attn.weight" in name:
                # Fused QKV — needs reshape before Muon update
                cattn_params.append(p)
            else:
                muon_params.append(p)
        else:
            adam_params.append(p)

    return muon_params, adam_params, cattn_params


# Import the core Muon functions from your existing muon.py
from muon import zeropower_via_newtonschulz5, muon_update, adam_update


def muon_update_cattn(grad, momentum, n_embd, beta=0.95, ns_steps=5, nesterov=True):
    """
    Muon update for fused c_attn weights: reshape (n_embd, 3*n_embd) -> (3, n_embd, n_embd),
    apply Newton-Schulz independently to Q, K, V, then reshape back.
    """
    # Reshape grad and momentum to (3, n_embd, n_embd)
    orig_shape = grad.shape  # (n_embd, 3*n_embd)
    grad_3d = grad.view(n_embd, 3, n_embd).permute(1, 0, 2).contiguous()       # (3, n_embd, n_embd)
    momentum_3d = momentum.view(n_embd, 3, n_embd).permute(1, 0, 2).contiguous()  # (3, n_embd, n_embd)

    # Standard Muon update on the batched (3, n_embd, n_embd) tensor
    momentum_3d.lerp_(grad_3d, 1 - beta)
    update_3d = grad_3d.lerp_(momentum_3d, beta) if nesterov else momentum_3d
    update_3d = zeropower_via_newtonschulz5(update_3d, steps=ns_steps)
    update_3d *= max(1, update_3d.size(-2) / update_3d.size(-1)) ** 0.5

    # Reshape back to (n_embd, 3*n_embd)
    update = update_3d.permute(1, 0, 2).contiguous().view(orig_shape)

    # Also reshape momentum back in-place
    momentum.copy_(momentum_3d.permute(1, 0, 2).contiguous().view(orig_shape))

    return update


class MuonWithAuxAdamFixed(torch.optim.Optimizer):
    """
    Fixed version of MuonWithAuxAdam that correctly handles HuggingFace GPT-2's
    fused c_attn (QKV) weight matrices.

    Usage:
        muon_params, adam_params, cattn_params = split_muon_params_fixed(model)

        param_groups = [
            dict(params=muon_params, use_muon=True, is_cattn=False,
                 lr=9e-2, weight_decay=0.1),
            dict(params=cattn_params, use_muon=True, is_cattn=True,
                 lr=9e-2, weight_decay=0.1, n_embd=1024),
            dict(params=adam_params, use_muon=False, is_cattn=False,
                 lr=6e-4, betas=(0.85, 0.999), weight_decay=0.1),
        ]
        optimizer = MuonWithAuxAdamFixed(param_groups)
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            group.setdefault("is_cattn", False)

            if group["use_muon"]:
                group["params"] = sorted(
                    group["params"], key=lambda x: x.size(), reverse=True
                )
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0)
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
                group.setdefault("weight_decay", 0)

        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"] and group["is_cattn"]:
                # --- Fused c_attn: reshape to (3, d, d) before Muon ---
                n_embd = group["n_embd"]
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    update = muon_update_cattn(
                        p.grad, state["momentum_buffer"],
                        n_embd=n_embd, beta=group["momentum"]
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

            elif group["use_muon"]:
                # --- Standard Muon (non-fused matrices) ---
                params = group["params"]
                if dist.is_initialized():
                    ws = dist.get_world_size()
                    rk = dist.get_rank()
                    params_pad = params + [torch.empty_like(params[-1])] * (
                        ws - len(params) % ws
                    )
                    for base_i in range(len(params))[::ws]:
                        if base_i + rk < len(params):
                            p = params[base_i + rk]
                            if p.grad is None:
                                p.grad = torch.zeros_like(p)
                            state = self.state[p]
                            if len(state) == 0:
                                state["momentum_buffer"] = torch.zeros_like(p)
                            upd = muon_update(
                                p.grad, state["momentum_buffer"],
                                beta=group["momentum"]
                            )
                            p.mul_(1 - group["lr"] * group["weight_decay"])
                            p.add_(upd.reshape(p.shape), alpha=-group["lr"])
                        dist.all_gather(
                            params_pad[base_i : base_i + ws],
                            params_pad[base_i + rk],
                        )
                else:
                    for p in params:
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        upd = muon_update(
                            p.grad, state["momentum_buffer"],
                            beta=group["momentum"]
                        )
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(upd.reshape(p.shape), alpha=-group["lr"])

            else:
                # --- AdamW group ---
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    upd = adam_update(
                        p.grad, state["exp_avg"], state["exp_avg_sq"],
                        state["step"], group["betas"], group["eps"]
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(upd, alpha=-group["lr"])

        return loss


# ---------------------------------------------------------------------------
# Example integration with train_muon_lm1b.py
# ---------------------------------------------------------------------------

EXAMPLE_USAGE = """
# In train_muon_lm1b.py, replace:
#
#   muon_params, adam_params = split_muon_params(model)
#   param_groups = [
#       dict(params=muon_params, use_muon=True, ...),
#       dict(params=adam_params, use_muon=False, ...),
#   ]
#   optimizer = MuonWithAuxAdam(param_groups)
#
# With:
#
#   from fix_muon_qkv import split_muon_params_fixed, MuonWithAuxAdamFixed
#
#   muon_params, adam_params, cattn_params = split_muon_params_fixed(model)
#
#   param_groups = [
#       dict(params=muon_params, use_muon=True, is_cattn=False,
#            lr=args.muon_lr, weight_decay=args.weight_decay),
#       dict(params=cattn_params, use_muon=True, is_cattn=True,
#            lr=args.muon_lr, weight_decay=args.weight_decay,
#            n_embd=args.n_embd),
#       dict(params=adam_params, use_muon=False, is_cattn=False,
#            lr=args.lr, betas=(args.adam_beta1, args.adam_beta2),
#            weight_decay=args.weight_decay),
#   ]
#   optimizer = MuonWithAuxAdamFixed(param_groups)
"""