"""
Step 3: Exoplanet Detection Project — Reusable Preprocessing Pipeline

Goal: turn "download + flatten + fold" (what we did manually for Kepler-10)
into ONE reusable function that takes (target_name, period, t0, duration)
and returns the global view + local view arrays that feed an Astronet-style
CNN. This is the function you'll call hundreds of times in Week 4 once you
have a labeled list of confirmed planets + false positives.

Run this locally:
    pip install lightkurve astropy numpy
    python step3_preprocessing_pipeline.py
"""

import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt

# Bin counts follow the original Astronet paper (Shallue & Vanderburg 2018)
GLOBAL_BINS = 2001
LOCAL_BINS = 201
# How many transit durations wide the local (zoomed-in) view should span
LOCAL_VIEW_DURATIONS = 4


def fetch_and_flatten(target_name, n_quarters=4, mission="Kepler"):
    """Download, stitch, and flatten a light curve. Returns a flattened lc."""
    print(f"[{target_name}] Searching for light curve data...")
    search_result = lk.search_lightcurve(target_name, mission=mission, cadence="long")
    if len(search_result) == 0:
        raise ValueError(f"No data found for {target_name}")

    n = min(n_quarters, len(search_result))
    print(f"[{target_name}] Downloading {n} quarters...")
    lc_collection = search_result[:n].download_all()
    lc = lc_collection.stitch().remove_nans()
    flat_lc = lc.flatten(window_length=401)
    return flat_lc


def bin_curve(phase, flux, num_bins, bin_width):
    """
    Bin a phase-folded curve into a fixed number of bins spanning
    [-bin_width/2, +bin_width/2]. Uses the median flux value within each
    bin, matching Astronet's approach, and fills empty bins by
    interpolation so every output always has exactly num_bins points
    (a CNN needs fixed-size input every time).
    """
    bin_edges = np.linspace(-bin_width / 2, bin_width / 2, num_bins + 1)
    binned = np.full(num_bins, np.nan)

    for i in range(num_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
        if mask.sum() > 0:
            binned[i] = np.median(flux[mask])

    # Fill any empty bins (gaps) by linear interpolation so the CNN always
    # gets a complete, fixed-length array
    nan_mask = np.isnan(binned)
    if nan_mask.any():
        valid_idx = np.where(~nan_mask)[0]
        binned[nan_mask] = np.interp(np.where(nan_mask)[0], valid_idx, binned[valid_idx])

    return binned


def make_global_local_views(flat_lc, period, t0, duration):
    """
    The core function: takes a flattened light curve + known/candidate
    transit parameters (period, t0, duration in days) and returns:
        global_view: fixed-length array (GLOBAL_BINS,) — the whole orbit
        local_view:  fixed-length array (LOCAL_BINS,)  — zoomed on the transit
    This is exactly the two-input representation Astronet's CNN expects.
    """
    folded = flat_lc.fold(period=period, epoch_time=t0)
    phase = folded.time.value  # phase in days, centered at 0 = transit center
    flux = folded.flux.value

    # Global view: spans the full period (one complete orbit cycle)
    global_view = bin_curve(phase, flux, GLOBAL_BINS, bin_width=period)

    # Local view: zoomed in tightly around the transit itself
    local_width = duration * LOCAL_VIEW_DURATIONS
    local_view = bin_curve(phase, flux, LOCAL_BINS, bin_width=local_width)

    # Normalize both views: center on median, scale so the min flux is -1
    # (standard Astronet preprocessing so depth is comparable across stars
    # of different brightness)
    for view in (global_view, local_view):
        view -= np.median(view)
        min_val = np.abs(view.min())
        if min_val > 0:
            view /= min_val

    return global_view, local_view


def main():
    # Kepler-10b parameters — the ones BLS already recovered for us in Step 2
    target = "Kepler-10"
    period = 0.8375
    t0 = 0.0  # using fold's own epoch since we already fold at best_t0 in step 2;
              # here we just re-derive it fresh via fold() for simplicity
    duration = 0.075  # ~1.8 hours, known duration for Kepler-10b

    flat_lc = fetch_and_flatten(target)

    print(f"\nBuilding global/local views for {target} at period={period} days...")
    global_view, local_view = make_global_local_views(flat_lc, period, t0, duration)

    print(f"Global view shape: {global_view.shape}")
    print(f"Local view shape:  {local_view.shape}")

    # Plot both views side by side — this is what will get fed to the CNN
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(global_view, lw=0.8)
    axes[0].set_title(f"{target} — Global View ({GLOBAL_BINS} bins)")
    axes[0].set_xlabel("Bin index")
    axes[0].set_ylabel("Normalized flux")

    axes[1].plot(local_view, lw=0.8, color="orange")
    axes[1].set_title(f"{target} — Local View ({LOCAL_BINS} bins, zoomed on transit)")
    axes[1].set_xlabel("Bin index")

    plt.tight_layout()
    plt.savefig("kepler10_global_local_views.png", dpi=150)
    print("\nSaved plot to kepler10_global_local_views.png")

    # Save the raw arrays too — this is the actual "training example" format
    # you'll be generating in bulk in Week 4
    np.savez("kepler10_example.npz", global_view=global_view, local_view=local_view)
    print("Saved arrays to kepler10_example.npz (this is your first training example!)")


if __name__ == "__main__":
    main()
