"""
Step 2: Exoplanet Detection Project — Box Least Squares (BLS) Search
Target: Kepler-10 (confirmed host of Kepler-10b)

Known values for Kepler-10b (from NASA Exoplanet Archive), used here only
to sanity-check that BLS recovers the right answer:
    Orbital period   ~ 0.837491 days
    Transit duration ~ 1.8 hours (~0.075 days)
    Transit depth    ~ 150-170 ppm

Run this locally:
    pip install lightkurve astropy numpy
    python step2_bls_search.py

Requires more data than a single quarter to get a reliable BLS result —
this script pulls multiple quarters and stitches them together.
"""

import numpy as np
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares
import matplotlib.pyplot as plt

TARGET = "Kepler-10"

# Known period range to search around (days). We search a wide range first
# since in a real pipeline you would NOT know the period in advance —
# this wide search demonstrates BLS actually finding it blind.
MIN_PERIOD = 0.5
MAX_PERIOD = 10.0


def main():
    print(f"Searching for all available Kepler light curves on {TARGET}...")
    search_result = lk.search_lightcurve(TARGET, mission="Kepler", cadence="long")
    print(search_result)

    if len(search_result) == 0:
        print("No results found.")
        return

    # Pull several quarters (more data = more transits = better BLS signal).
    # Downloading all of them can be slow; start with the first 4 quarters.
    n_quarters = min(4, len(search_result))
    print(f"\nDownloading {n_quarters} quarters of data (this may take a minute)...")
    lc_collection = search_result[:n_quarters].download_all()

    # Stitch quarters into one continuous light curve and flatten it
    lc = lc_collection.stitch()
    lc = lc.remove_nans()
    flat_lc = lc.flatten(window_length=401)

    print(f"\nTotal data points after stitching: {len(flat_lc.flux)}")
    baseline_days = flat_lc.time.value.max() - flat_lc.time.value.min()
    print(f"Time baseline: {baseline_days:.1f} days")

    # --- Run BLS ---
    print(f"\nRunning Box Least Squares search from {MIN_PERIOD} to {MAX_PERIOD} days...")
    bls = BoxLeastSquares(flat_lc.time.value, flat_lc.flux.value)

    # Duration grid: candidate transit durations to test, in days
    durations = np.linspace(0.01, 0.2, 20)
    periodogram = bls.autopower(durations, minimum_period=MIN_PERIOD, maximum_period=MAX_PERIOD)

    best_period = periodogram.period[np.argmax(periodogram.power)]
    best_t0 = periodogram.transit_time[np.argmax(periodogram.power)]
    best_duration = periodogram.duration[np.argmax(periodogram.power)]

    print(f"\n--- BLS Result ---")
    print(f"Best period found:   {best_period:.6f} days")
    print(f"Known period:        0.837491 days (Kepler-10b)")
    print(f"Best transit time:   {best_t0:.4f}")
    print(f"Best duration:       {best_duration:.4f} days")

    # --- Plot 1: periodogram (power vs period) ---
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(periodogram.period, periodogram.power, lw=0.5)
    ax.axvline(best_period, color="red", linestyle="--", alpha=0.7, label=f"Best period: {best_period:.4f}d")
    ax.set_xlabel("Period [days]")
    ax.set_ylabel("BLS Power")
    ax.set_title(f"{TARGET} — BLS Periodogram")
    ax.legend()
    plt.tight_layout()
    plt.savefig("kepler10_bls_periodogram.png", dpi=150)
    print("\nSaved plot to kepler10_bls_periodogram.png")

    # --- Plot 2: phase-folded light curve at best period ---
    folded_lc = flat_lc.fold(period=best_period, epoch_time=best_t0)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    folded_lc.scatter(ax=ax2, s=2, alpha=0.4)
    folded_lc.bin(time_bin_size=0.005).plot(ax=ax2, color="red", lw=1.5, label="Binned")
    ax2.set_title(f"{TARGET} — Phase-Folded at {best_period:.4f} days")
    ax2.legend()
    plt.tight_layout()
    plt.savefig("kepler10_phase_folded.png", dpi=150)
    print("Saved plot to kepler10_phase_folded.png")


if __name__ == "__main__":
    main()
