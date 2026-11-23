"""Shared plotting helpers for the example notebooks.

Plotting lives here rather than in ``flow_inference`` so that the library has no
matplotlib dependency: the library computes fields and the notebooks draw them.

To use from a notebook in any ``examples/`` subdirectory::

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path.cwd().parent))
    from plotting import plot_band, plot_traces, show_overlay

There are two families of helper:

* **1D paths and functions**: :func:`plot_band` (median and quantile band) and
  :func:`plot_traces` (individual sample paths), combined in :func:`plot_paths`.
  Pass ``channel`` to select a channel of a multi-channel process, and ``x`` to
  supply a time grid (``linspace(0, 1, N)`` by default).
* **2D fields on a grid**: :func:`show_overlay` (a row of panels with optional
  markers, region boundaries and shared colour limits), :func:`show_fields` (a
  numbered row of samples) and :func:`shared_range` (common colour limits for
  several calls).
"""

import matplotlib.pyplot as plt
import torch
from matplotlib.markers import MarkerStyle


def as_numpy(x):
    """Convert a tensor to a numpy array; arrays and lists pass through unchanged."""
    return x.detach().cpu().numpy() if torch.is_tensor(x) else x


def _paths_and_x(samples, x=None, channel=None):
    """Reduce ``samples`` to ``(B, N)`` paths and build the matching x-grid.

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
    """Plot the median path and a shaded quantile band across the samples.

    Args:
        samples: paths ``(B, N)``, ``(B, 1, N)`` or ``(B, C, N)`` (with ``channel``).
        x: the x-grid ``(N,)``; ``linspace(0, 1, N)`` by default.
        channel: which channel to plot; required when ``C > 1``.
        color, label, alpha, linewidth: styling of the band and median line.
        ci: lower and upper quantiles of the band.
        ax: axis to draw on; the current axis by default.

    Returns:
        ``(ax, x)``, so the caller can layer notebook-specific overlays (observed
        points, bound curves, true trajectories) on the same axis and grid.
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

    Pair with :func:`plot_band` for the usual band-and-members figure.

    Args:
        samples, x, channel, ax: as for :func:`plot_band`.
        count: the number of paths drawn (the first ``count`` samples).
        color, alpha, linewidth, zorder: line styling.
        label: legend label, attached to the first path only.

    Returns:
        ``(ax, x)``.
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
    """Plot a quantile band and a few sample paths: the common 1D summary figure.

    A thin wrapper over :func:`plot_band` and :func:`plot_traces` that also labels
    the axes.

    Args:
        samples, x, ax: as for :func:`plot_band`.
        num_samples: the number of individual paths drawn.
        ci: lower and upper quantiles of the band.
        show_band: whether to draw the band.

    Returns:
        ``(ax, x)``.
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


def shared_range(*tensors, pad=0.0):
    """Common ``(vmin, vmax)`` across all given tensors, for a shared colour scale.

    The tensors may have different sizes, e.g. a coarse target and a fine posterior.
    ``pad`` widens the range on each side by that fraction of its span.
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
    """Plot fields on an ``(h, w)`` grid as side-by-side panels with optional overlays.

    Args:
        field: tensor reshapeable to ``(-1, h, w)``, one panel per field.
        h, w: the grid shape.
        n: show at most this many panels; all by default.
        titles: per-panel titles; ``1, 2, ...`` by default.
        markers: dicts, each drawn on every panel::

                {"flat_idx": LongTensor of flat grid indices,
                 "marker": matplotlib marker (e.g. "^", "v", "x"),
                 "color": face colour,
                 "label": legend label}

            with optional ``"s"`` (size), ``"facecolor"`` and ``"edgecolor"``
            entries. Flat index ``f`` is grid cell ``(f // w, f % w)``.
        boundaries: ``(h, w)`` integer label map; thin white lines are drawn
            between cells whose labels differ (region boundaries).
        cmap: the colour map.
        suptitle: a title above the whole row.
        vmin, vmax: colour limits for every panel and its colourbar. Pass the same
            pair (e.g. from :func:`shared_range`) to several calls to put truth,
            observations and posterior on one scale.
    """
    fields = field.detach().reshape(-1, h, w)
    if n is not None:
        fields = fields[:n]
    fields = fields.cpu().numpy()
    n_panels = len(fields)
    _, axes = plt.subplots(1, n_panels, figsize=(2.4 * n_panels, 2.4))
    if n_panels == 1:
        axes = [axes]

    # Edges between cells with differing labels, shared by every panel.
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
    """Plot 2D samples as a numbered row of panels.

    A thin wrapper over :func:`show_overlay`.

    Args:
        samples: ``(B, C, H, W)`` (the first channel is shown), ``(B, H, W)``, or
            ``(B, N)`` with ``h`` and ``w`` given.
        h, w: the grid shape, needed only for ``(B, N)`` samples.
        n: show at most this many panels; all by default.
        cmap: the colour map.
        subtitle: if given, panel titles read ``"{subtitle} 1"``, ``"{subtitle} 2"``,
            and so on.
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
