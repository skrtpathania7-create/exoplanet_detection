"""
Step 4: Exoplanet Detection Project — Build a Labeled Training Dataset

Goal: pull a list of known CONFIRMED planets and known FALSE POSITIVEs from
NASA's Kepler Objects of Interest (KOI) catalog, then run the Step 3
preprocessing pipeline (global/local views) on each one, saving everything
into a single dataset file ready for CNN training.

This is the biggest, slowest step so far — it downloads light curves for
many different stars. Expect it to take a while depending on how many
targets you set below. Start small (N_PER_CLASS = 15-25) to confirm it
works end-to-end before scaling up.

Run this locally:
    pip install lightkurve astropy numpy pandas
    python step4_build_dataset.py
"""

import time
import urllib.parse
import numpy as np
import pandas as pd
import lightkurve as lk

# --- Config ---
N_PER_CLASS = 120      # how many CONFIRMED and how many FALSE POSITIVE targets to pull
                        # (raised from 20 — more data needed to avoid the model
                        # taking the "just guess confirmed" shortcut we saw)
GLOBAL_BINS = 2001
LOCAL_BINS = 201
LOCAL_VIEW_DURATIONS = 4
OUTPUT_FILE = "exoplanet_dataset_large.npz"  # renamed so your first 40-example
                                              # run isn't overwritten


def fetch_koi_catalog():
    """
    Query NASA's Exoplanet Archive (KOI cumulative table) for a list of
    confirmed planets and false positives, with the orbital parameters
    (period, epoch, duration) we need to fold each light curve.
    """
    print("Querying NASA Exoplanet Archive for KOI catalog...")
    query = (
        "select kepid,koi_disposition,koi_period,koi_time0bk,koi_duration "
        "from cumulative "
        "where koi_disposition in ('CONFIRMED','FALSE POSITIVE')"
    )
    url = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
        + urllib.parse.quote(query)
        + "&format=csv"
    )
    df = pd.read_csv(url)
    df = df.dropna(subset=["koi_period", "koi_time0bk", "koi_duration"])
    print(f"Retrieved {len(df)} labeled KOIs total "
          f"({(df.koi_disposition == 'CONFIRMED').sum()} confirmed, "
          f"{(df.koi_disposition == 'FALSE POSITIVE').sum()} false positives)")
    return df


def select_balanced_sample(df, n_per_class):
    """Take a random, balanced sample of N confirmed + N false positive KOIs."""
    confirmed = df[df.koi_disposition == "CONFIRMED"].sample(
        n=min(n_per_class, (df.koi_disposition == "CONFIRMED").sum()), random_state=42
    )
    false_pos = df[df.koi_disposition == "FALSE POSITIVE"].sample(
        n=min(n_per_class, (df.koi_disposition == "FALSE POSITIVE").sum()), random_state=42
    )
    sample = pd.concat([confirmed, false_pos]).reset_index(drop=True)
    print(f"Selected {len(confirmed)} confirmed + {len(false_pos)} false positives = "
          f"{len(sample)} targets for this run")
    return sample


def bin_curve(phase, flux, num_bins, bin_width):
    """Bin a phase-folded curve into a fixed number of bins (same as Step 3)."""
    bin_edges = np.linspace(-bin_width / 2, bin_width / 2, num_bins + 1)
    binned = np.full(num_bins, np.nan)
    for i in range(num_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
        if mask.sum() > 0:
            binned[i] = np.median(flux[mask])
    nan_mask = np.isnan(binned)
    if nan_mask.any():
        valid_idx = np.where(~nan_mask)[0]
        if len(valid_idx) < 2:
            return None  # too sparse to interpolate, discard this example
        binned[nan_mask] = np.interp(np.where(nan_mask)[0], valid_idx, binned[valid_idx])
    return binned


def make_global_local_views(flat_lc, period, t0, duration):
    """Same as Step 3 — produces (global_view, local_view) or (None, None) on failure."""
    folded = flat_lc.fold(period=period, epoch_time=t0)
    phase = folded.time.value
    flux = folded.flux.value

    global_view = bin_curve(phase, flux, GLOBAL_BINS, bin_width=period)
    local_width = duration / 24.0 * LOCAL_VIEW_DURATIONS  # koi_duration is in HOURS
    local_view = bin_curve(phase, flux, LOCAL_BINS, bin_width=local_width)

    if global_view is None or local_view is None:
        return None, None

    for view in (global_view, local_view):
        view -= np.median(view)
        min_val = np.abs(view.min())
        if min_val > 0:
            view /= min_val

    return global_view, local_view


def process_one_target(kepid, period, t0, duration_hours):
    """Download + flatten + fold one target. Returns (global, local) or (None, None)."""
    try:
        search_result = lk.search_lightcurve(f"KIC {kepid}", mission="Kepler", cadence="long")
        if len(search_result) == 0:
            return None, None
        # Use up to 4 quarters to keep runtime reasonable across many targets
        n = min(4, len(search_result))
        lc_collection = search_result[:n].download_all()
        lc = lc_collection.stitch().remove_nans()
        flat_lc = lc.flatten(window_length=401)
        return make_global_local_views(flat_lc, period, t0, duration_hours)
    except Exception as e:
        print(f"    [skipped KIC {kepid}: {type(e).__name__}: {e}]")
        return None, None


def main():
    koi_df = fetch_koi_catalog()
    sample = select_balanced_sample(koi_df, N_PER_CLASS)

    X_global, X_local, y, kepids_used = [], [], [], []

    start = time.time()
    for i, row in sample.iterrows():
        label = 1 if row.koi_disposition == "CONFIRMED" else 0
        print(f"[{i+1}/{len(sample)}] KIC {row.kepid} "
              f"({row.koi_disposition}, period={row.koi_period:.3f}d)...")

        global_view, local_view = process_one_target(
            row.kepid, row.koi_period, row.koi_time0bk, row.koi_duration
        )

        if global_view is not None:
            X_global.append(global_view)
            X_local.append(local_view)
            y.append(label)
            kepids_used.append(row.kepid)
        # else: silently skipped (logged inside process_one_target)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} minutes.")
    print(f"Successfully built {len(y)} examples out of {len(sample)} attempted "
          f"({sum(y)} confirmed, {len(y) - sum(y)} false positives)")

    if len(y) == 0:
        print("No examples were successfully built — nothing to save.")
        return

    np.savez(
        OUTPUT_FILE,
        X_global=np.array(X_global),
        X_local=np.array(X_local),
        y=np.array(y),
        kepids=np.array(kepids_used),
    )
    print(f"Saved dataset to {OUTPUT_FILE}")
    print(f"  X_global shape: {np.array(X_global).shape}")
    print(f"  X_local shape:  {np.array(X_local).shape}")
    print(f"  y shape:        {np.array(y).shape}")


if __name__ == "__main__":
    main()
