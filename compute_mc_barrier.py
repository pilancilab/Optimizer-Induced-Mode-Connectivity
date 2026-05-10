import json, numpy as np

with open("merger_eval_results/loss_interp_mean_err.json") as f:
    J = json.load(f)

lams = np.array(J["coeffs"], dtype=float)

def barrier_from_curve(losses):
    losses = np.array(losses, dtype=float)
    # endpoints: λ=1 is Θ_A, λ=0 is π(Θ_B) (learned) or Θ_B (vanilla)
    L_A = losses[np.argmax(lams)]   # λ=1
    L_B = losses[np.argmin(lams)]   # λ=0
    # print(f'{L_A=} {L_B=}')
    baseline = lams * L_A + (1 - lams) * L_B
    return float(np.max(losses - baseline))

learned_barriers = np.array([barrier_from_curve(c) for c in J["learned_all"]], dtype=float)
vanilla_barriers = np.array([barrier_from_curve(c) for c in J["vanilla_all"]], dtype=float)

print("Learned barrier mean±std:", learned_barriers.mean(), learned_barriers.std(ddof=1))
print("Vanilla barrier mean±std:", vanilla_barriers.mean(), vanilla_barriers.std(ddof=1))
