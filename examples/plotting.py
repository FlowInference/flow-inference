"""Shared plotting helpers for the example notebooks.

Plotting lives here rather than in ``flow_inference`` so the library stays free of a
matplotlib dependency — it computes fields, the notebooks draw them.

To use from a notebook in any ``examples/`` subdirectory::

    import sys; sys.path.insert(0, str(Path.cwd().parent))
    from plotting import plot_band, plot_traces, show_overlay

Two families:

* **1D paths / functions** — :func:`plot_band` (quantile band + median) and
  :func:`plot_traces` (individual sample lines). Every SDE and GP notebook builds its
  path figures from these two; pass ``channel`` to select a channel of a multi-channel
  process, and ``x`` to supply a time grid (defaults to ``linspace(0, 1, N)``).
* **2D fields on a grid** — :func:`show_overlay` (panel row with optional markers,
  region boundaries and shared colour limits) and :func:`shared_range` for putting
  several calls on one colour scale.
* **Guidance diagnostics** — :func:`plot_guidance_norms` renders the per-step history
  a guidance object records when ``record_history=True``, showing where the tanh
  squash in ``clip_vector_field`` starts attenuating the drift.
"""

import matplotlib.pyplot as plt
import torch
from matplotlib.markers import MarkerStyle


def as_numpy(x):
    """Detach a tensor to a numpy array (a no-op passthrough for arrays/lists)."""
    return x.detach().cpu().numpy() if torch.is_tensor(x) else x


def _paths_and_x(samples, x=None, channel=None):
    """Normalise ``samples`` to ``(B, N)`` and build the matching x-grid.

    Accepts ``(B, N)``, ``(B, 1, N)`` or ``(B, C, N)``; ``channel`` selects a channel
    of a multi-channel process (required when ``C > 1``). ``x`` defaults to
    ``linspace(0, 1, N)``.
    """
    if samples.ndim == 3:
        if channel is not None:
            samples = samples[:, channel]
        elif samples.shape[1] == 1:
            samples = samples[:, 0]
        else:
            raise ValueError(
                f"samples has {samples.shape[1]} channels — pass channel=... "
                "to select which one to plot"
            )
    paths = samples  # (B, N)
    N = paths.shape[-1]
    xs = torch.linspace(0, 1, N) if x is None else x
    return paths, as_numpy(xs)


def plot_band(
    samples,
    x=None,
    channel=None,
    color="steelblue",
    label=None,
    ci=(0.1, 0.9),
    alpha=0.18,
    linewidth=2.5,
    ax=None,
):
    """Median line plus a shaded ``ci`` quantile band across the sample dimension.

    ``samples`` is ``(B, N)``, ``(B, 1, N)`` or ``(B, C, N)`` (with ``channel``).
    Returns ``(ax, x)`` so the caller can layer notebook-specific overlays — observed
    points, bound curves, truth trajectories — on the same axis and time grid.
    """
    paths, xs = _paths_and_x(samples, x, channel)
    ax = ax or plt.gca()

    levels = torch.tensor([ci[0], 0.5, ci[1]], device=paths.device, dtype=paths.dtype)
    q_lo, q_mid, q_hi = paths.quantile(levels, dim=0)
    ax.fill_between(
        xs, as_numpy(q_lo), as_numpy(q_hi), color=color, alpha=alpha, linewidth=0
    )
    ax.plot(xs, as_numpy(q_mid), color=color, linewidth=linewidth, label=label)
    return ax, xs


def plot_traces(
    samples,
    x=None,
    channel=None,
    color="steelblue",
    count=16,
    alpha=0.12,
    linewidth=0.9,
    label=None,
    ax=None,
    zorder=None,
):
    """Overlay up to ``count`` individual sample paths.

    Same input shapes as :func:`plot_band`; pair the two to get the usual
    band-plus-members figure. Returns ``(ax, x)``.
    """
    paths, xs = _paths_and_x(samples, x, channel)
    ax = ax or plt.gca()

    for i, path in enumerate(paths[:count]):
        ax.plot(
            xs,
            as_numpy(path),
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            label=label if i == 0 else None,
            zorder=zorder,
        )
    return ax, xs


def plot_paths(
    samples, num_samples=5, ci=(0.05, 0.95), show_band=True, ax=None, x=None
):
    """Band + a few member traces in one call — the common 1D summary figure.

    Thin wrapper over :func:`plot_band` and :func:`plot_traces`, kept because most GP
    notebooks want exactly this. Returns ``(ax, x)``.
    """
    ax = ax or plt.gca()
    if show_band:
        ax, xs = plot_band(samples, x=x, ci=ci, ax=ax, linewidth=1.5, alpha=0.5)
    ax, xs = plot_traces(
        samples, x=x, count=num_samples, alpha=0.5, linewidth=1.0, ax=ax
    )
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    return ax, xs


def guidance_summary(guidance, samples=None, latent_dim=None):
    """Scalar summary of a recorded guidance run — the numbers worth reading first.

    Returns a dict with:

      ``arc_length``    Σ dt·‖dv‖ over the trajectory. With the squash saturated this
                        equals ``max_norm`` almost exactly, because ‖dv‖ is pinned
                        there and the sampler integrates over Δt = 1. Read `max_norm`
                        as a total path-length budget, not a rarely-hit cap.
      ``saturated_frac`` fraction of steps with mean attenuation < 0.5 — i.e. where the
                        gradient magnitude (and so the condition's sigma) is being
                        thrown away and only the direction survives.
      ``reversal_frac`` fraction of steps whose direction flipped (persistence < 0).
                        High values mean the flow is chattering across the constraint
                        and spending budget without net displacement.
      ``ess_*``         importance-weight effective sample size (NaN for non-MC
                        guidance). ``ess_min_mean`` near 1 with S > 1 means the weights
                        have collapsed and the estimator is really doing DPS.
      ``shell_ratio``   if ``samples`` and ``latent_dim`` are given, ‖z‖ / sqrt(N).
                        Runs that land near 1 stayed inside the prior's typical set;
                        markedly above means the path budget overshot.
    """
    history = getattr(guidance, "history", guidance)
    if not history:
        raise ValueError("no recorded history — set guidance.record_history = True")

    ts = [r["t"] for r in history]
    arc = 0.0
    for i, r in enumerate(history):
        # dt from consecutive recorded times; the last step reuses the previous gap.
        if i + 1 < len(ts):
            dt = abs(ts[i] - ts[i + 1])
        else:
            dt = abs(ts[-2] - ts[-1]) if len(ts) > 1 else 0.0
        arc += dt * r["out_norm_mean"]

    atten = [r["attenuation_mean"] for r in history]
    persist = [
        r["persistence"] for r in history if r["persistence"] == r["persistence"]
    ]
    ess_mean = [r["ess_mean"] for r in history if r["ess_mean"] == r["ess_mean"]]
    ess_min = [r["ess_min"] for r in history if r["ess_min"] == r["ess_min"]]

    out = {
        "steps": len(history),
        "max_norm": history[0]["max_norm"],
        "arc_length": arc,
        "peak_raw_norm": max(r["raw_norm_max"] for r in history),
        "saturated_frac": sum(a < 0.5 for a in atten) / len(atten),
        "attenuation_mean": sum(atten) / len(atten),
        "reversal_frac": (
            sum(p < 0 for p in persist) / len(persist) if persist else float("nan")
        ),
        "num_mc_samples": history[0]["num_mc_samples"],
        "ess_mean": sum(ess_mean) / len(ess_mean) if ess_mean else float("nan"),
        "ess_min_mean": sum(ess_min) / len(ess_min) if ess_min else float("nan"),
    }
    if samples is not None and latent_dim is not None:
        norm = samples.detach().reshape(samples.shape[0], -1).norm(dim=-1).mean().item()
        out["shell_ratio"] = norm / (latent_dim**0.5)
    return out


def print_guidance_summary(guidance, samples=None, latent_dim=None):
    """Human-readable :func:`guidance_summary`, with the caveats spelled out."""
    s = guidance_summary(guidance, samples, latent_dim)
    print(f"steps                 : {s['steps']}")
    print(f"max_norm (budget)     : {s['max_norm']}")
    print(
        f"arc length  Σ dt‖dv‖  : {s['arc_length']:.2f}   <- ≈ max_norm when saturated"
    )
    print(f"peak raw ‖dv‖         : {s['peak_raw_norm']:.3e}")
    print(
        f"mean attenuation      : {s['attenuation_mean']:.4f}"
        f"   (1 = untouched, →0 = direction-only)"
    )
    print(f"steps saturated (<0.5): {s['saturated_frac']:.3f}")
    print(f"steps reversing       : {s['reversal_frac']:.3f}   <- high = chattering")
    if s["num_mc_samples"] is not None:
        print(f"num_mc_samples        : {s['num_mc_samples']}")
        print(
            f"ESS mean / min        : {s['ess_mean']:.2f} / {s['ess_min_mean']:.2f}"
            f"   <- ~1 means weights collapsed to DPS"
        )
    if "shell_ratio" in s:
        print(
            f"‖z‖ / sqrt(N)         : {s['shell_ratio']:.3f}"
            f"   <- ≈1 stayed in the prior's typical set"
        )
    if s["saturated_frac"] > 0.9:
        print(
            "\nNOTE: the squash is saturated for >90% of steps, so the condition's"
            "\n      sigma is divided out and has no effect. max_norm is the only"
            "\n      live magnitude knob in this regime."
        )
    return s


def plot_guidance_norms(guidance, logy=True, figsize=(11, 3.4), axes=None):
    """Plot recorded guidance magnitudes and where the tanh squash bites.

    Pass a guidance object that was run with ``record_history = True`` (or its
    ``.history`` list directly). Three panels, all against integration time t with the
    axis reversed so it reads left-to-right in sampling order (t=1 noise → t=0 data):

      1. **Magnitude** — pre-squash ‖dv‖ (mean, with min–max band) against the
         ``max_norm`` threshold. Where the band sits above the dashed line, the squash
         is active.
      2. **Attenuation** — the factor actually applied, ``tanh(r)/r`` with
         ``r = ‖dv‖/max_norm``. 1.0 means untouched; small values mean saturated, so
         the guidance direction survives but its magnitude is being discarded.
      3. **Fraction over threshold** — share of (sample, channel) vectors above
         ``max_norm`` at each step.

    Note the squash is *smooth*, not a hard clip: there is no step change at
    ``max_norm``. Attenuation is already ~0.76 at ``‖dv‖ = max_norm`` and falls off
    like ``1/r`` beyond it, so panel 2 is the honest picture of what is being lost.

    Returns the array of axes.
    """
    history = getattr(guidance, "history", guidance)
    if not history:
        raise ValueError(
            "no recorded history — set guidance.record_history = True before "
            "sampling (and call guidance.reset_history() between runs)"
        )

    t = [r["t"] for r in history]
    raw_mean = [r["raw_norm_mean"] for r in history]
    raw_max = [r["raw_norm_max"] for r in history]
    raw_min = [r["raw_norm_min"] for r in history]
    out_mean = [r["out_norm_mean"] for r in history]
    atten_mean = [r["attenuation_mean"] for r in history]
    atten_min = [r["attenuation_min"] for r in history]
    frac = [r["frac_over_max"] for r in history]
    max_norm = history[0]["max_norm"]

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=figsize)

    ax = axes[0]
    ax.fill_between(t, raw_min, raw_max, alpha=0.2, color="steelblue", linewidth=0)
    ax.plot(t, raw_mean, color="steelblue", lw=2, label="‖dv‖ pre-squash")
    ax.plot(t, out_mean, color="seagreen", lw=1.6, ls="-", label="‖dv‖ post-squash")
    if max_norm is not None:
        ax.axhline(max_norm, color="crimson", ls="--", lw=1.3, label="max_norm")
    ax.set_ylabel("guidance norm")
    ax.set_title("magnitude", fontsize=10)
    if logy:
        ax.set_yscale("log")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    ax.fill_between(
        t, atten_min, atten_mean, alpha=0.2, color="darkorange", linewidth=0
    )
    ax.plot(t, atten_mean, color="darkorange", lw=2, label="mean")
    ax.plot(t, atten_min, color="darkorange", lw=1.0, ls=":", label="worst case")
    ax.axhline(1.0, color="grey", ls="--", lw=1.0)
    ax.set_ylim(-0.03, 1.08)
    ax.set_ylabel("tanh(r)/r")
    ax.set_title("attenuation applied (1 = untouched)", fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[2]
    ax.plot(t, frac, color="crimson", lw=2)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("fraction")
    ax.set_title(f"vectors over max_norm={max_norm}", fontsize=10)

    for ax in axes:
        ax.set_xlabel("t  (1 = noise → 0 = data)")
        ax.invert_xaxis()
        ax.grid(True, color="#e5e7eb", lw=0.7)
    plt.tight_layout()
    return axes


def shared_range(*tensors, pad=0.0):
    """Common ``(vmin, vmax)`` across all given tensors, so panels share a colour scale.

    Size-agnostic: tensors may live on different grids (e.g. a coarse target and a fine
    posterior). ``pad`` expands the range by a fraction of its span on each side.
    """
    lo = min(t.detach().min().item() for t in tensors)
    hi = max(t.detach().max().item() for t in tensors)
    if pad:
        span = (hi - lo) or 1.0
        lo, hi = lo - pad * span, hi + pad * span
    return lo, hi


def show_overlay(
    field,
    h,
    w,
    n=None,
    titles=None,
    markers=None,
    boundaries=None,
    cmap="viridis",
    suptitle=None,
    vmin=None,
    vmax=None,
):
    """Plot fields on the ``(h, w)`` grid as side-by-side panels with optional overlays.

    Parameters
    ----------
    field        : tensor reshapeable to ``(-1, h, w)``; up to ``n`` panels shown
                   (``n=None`` shows all).
    titles       : optional list of per-panel titles (defaults to ``1, 2, ...``).
    markers      : optional list of dicts, each drawn on *every* panel::

                     {"flat_idx": LongTensor of flat grid indices,
                      "marker":   matplotlib marker (e.g. "^", "v", "x"),
                      "color":    edge/face colour,
                      "label":    legend label}

                   flat index ``f`` maps to ``(row, col) = (f // w, f % w)``.
    boundaries   : optional ``(h, w)`` integer label map; thin white lines are drawn
                   between cells whose labels differ (region/segment boundaries).
    vmin, vmax   : optional shared colour limits applied to every panel (and its
                   colourbar). Pass the same pair (e.g. from :func:`shared_range`) to
                   several calls to put truth / observed / posterior on one scale.
    """
    fields = field.detach().reshape(-1, h, w)
    if n is not None:
        fields = fields[:n]
    fields = fields.cpu().numpy()
    n_panels = len(fields)
    _, axes = plt.subplots(1, n_panels, figsize=(2.4 * n_panels, 2.4))
    if n_panels == 1:
        axes = [axes]

    # Precompute boundary edges (between differing labels) once.
    seg_v = seg_h = None
    if boundaries is not None:
        b = boundaries.detach().reshape(h, w).cpu().numpy()
        seg_v = b[:, :-1] != b[:, 1:]  # vertical edges between (c, c+1)
        seg_h = b[:-1, :] != b[1:, :]  # horizontal edges between (r, r+1)

    for idx, ax in enumerate(axes):
        im = ax.imshow(fields[idx], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(
            titles[idx] if titles and idx < len(titles) else f"{idx + 1}", fontsize=9
        )
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if boundaries is not None:
            for r in range(h):
                for c in range(w - 1):
                    if seg_v[r, c]:
                        ax.plot(
                            [c + 0.5, c + 0.5],
                            [r - 0.5, r + 0.5],
                            color="white",
                            lw=0.8,
                        )
            for r in range(h - 1):
                for c in range(w):
                    if seg_h[r, c]:
                        ax.plot(
                            [c - 0.5, c + 0.5],
                            [r + 0.5, r + 0.5],
                            color="white",
                            lw=0.8,
                        )

        if markers:
            for m in markers:
                fi = m["flat_idx"].detach().cpu().numpy()
                rows, cols = fi // w, fi % w
                # Unfilled markers ("+", "x", "_") are drawn in their face colour only.
                filled = MarkerStyle(m["marker"]).is_filled()
                ax.scatter(
                    cols,
                    rows,
                    marker=m["marker"],
                    s=m.get("s", 36),
                    facecolors=m.get("facecolor", m["color"]),
                    edgecolors=m.get("edgecolor", "white") if filled else None,
                    linewidths=0.6 if filled else 1.2,
                    label=m["label"] if idx == 0 else None,
                    zorder=3,
                )
    if markers and any(m.get("label") for m in markers):
        axes[0].legend(
            loc="upper left", fontsize=6, framealpha=0.7, bbox_to_anchor=(0, 1.0)
        )
    if suptitle:
        plt.suptitle(suptitle, fontsize=10, y=1.02)
    plt.tight_layout()
    plt.show()


def show_fields(samples, h=None, w=None, n=None, cmap="viridis", subtitle=""):
    """Panel row of 2D fields — the replacement for the old ``utils.plot_samples(dim=2)``.

    ``samples`` is ``(B, C, H, W)`` (channel squeezed) or ``(B, N)`` with ``h``/``w``
    given. A thin wrapper over :func:`show_overlay` with per-panel numbering.
    """
    if samples.ndim == 4:
        samples = samples[:, 0]
    if samples.ndim == 3:
        h, w = samples.shape[-2:]
    titles = None
    if subtitle:
        n_panels = len(samples) if n is None else min(n, len(samples))
        titles = [f"{subtitle} {i + 1}" for i in range(n_panels)]
    show_overlay(samples, h, w, n=n, titles=titles, cmap=cmap)
