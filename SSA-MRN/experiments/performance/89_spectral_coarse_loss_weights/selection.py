"""89 selection rule (config.json selection_rule), validation only. choose(stage, val) with val[sensor][x] = dict(psnr, sam) of the
screening-selected epoch (minimum validation MSE) returns the chosen value and the full trace. Pure Python, no torch."""
import math


def _ref(stage, conf, lam_star=None):
    return 0.46 if stage == 'A' else 1.0


def choose(stage, conf, val):
    st = conf['stage_' + stage]; rule = conf['selection_rule']; tau = rule['tau_sam_deg']; ref = _ref(stage, conf)
    sensors = ('QB', 'GF2', 'WV3')
    for s in sensors:
        if ref not in val[s]: raise ValueError(f'reference point {ref} missing on {s}')
        for x, v in val[s].items():
            if not (math.isfinite(v['psnr']) and math.isfinite(v['sam'])): raise ValueError(f'nonfinite validation {s} {x}')
    delta = {s: {x: v['psnr'] - val[s][ref]['psnr'] for x, v in val[s].items()} for s in sensors}
    dsam = {s: {x: v['sam'] - val[s][ref]['sam'] for x, v in val[s].items()} for s in sensors}
    grid = sorted(float(x) for x in st['grid']['QB']); centers = [float(x) for x in st['center_candidates']]
    for x in grid:
        if x not in val['QB']: raise ValueError(f'QB grid point {x} missing')
    m = {}
    for i in range(1, len(grid) - 1):
        if grid[i] in centers: m[grid[i]] = (delta['QB'][grid[i - 1]] + delta['QB'][grid[i]] + delta['QB'][grid[i + 1]]) / 3
    x_qb = max(m, key=lambda x: (m[x], -x))
    shared = sorted(float(x) for x in st['shared_points'])
    for s in sensors:
        for p in shared:
            if p not in val[s]: raise ValueError(f'shared point {p} missing on {s}')
    smean = {p: sum(delta[s][p] for s in sensors) / 3 for p in shared}
    p_star = max(smean, key=lambda p: (smean[p], -p)); k = shared.index(p_star)
    lo, hi = shared[max(k - 1, 0)], shared[min(k + 1, len(shared) - 1)]
    first = x_qb if lo <= x_qb <= hi else p_star
    order = [first] + [x for x in sorted(m, key=lambda x: (-m[x], x)) if lo <= x <= hi and x != first] + [p for p in sorted(smean, key=lambda p: (-smean[p], p)) if p != first]
    order = list(dict.fromkeys(order)); checks = []; chosen = None
    for x in order:
        bad = {s: dsam[s][x] for s in sensors if x in dsam[s] and dsam[s][x] > tau[s]}
        checks.append(dict(value=x, sam_delta={s: dsam[s][x] for s in sensors if x in dsam[s]}, rejected_by=bad))
        if not bad: chosen = x; break
    fallback = chosen is None
    if fallback: chosen = ref
    edges = {centers[0], centers[-1], shared[0], shared[-1]}
    return dict(stage=stage, reference=ref, delta_psnr=delta, delta_sam=dsam, qb_moving_average=m, x_qb=x_qb, shared_mean=smean, p_star=p_star,
                p_star_interval=[lo, hi], x_qb_inside_interval=lo <= x_qb <= hi, candidate_order=order, sam_checks=checks, chosen=chosen,
                fell_back_to_reference=fallback, at_boundary=chosen in edges and not fallback)
