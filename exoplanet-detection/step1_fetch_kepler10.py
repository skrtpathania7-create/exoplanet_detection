"""
Step 1: Exoplanet Detection Project — First Light Curve Pull
Target: Kepler-10 (confirmed host of Kepler-10b, a rocky super-Earth)

Run this locally (not in a sandboxed env without internet access to MAST):
    pip install lightkurve astropy
    python step1_fetch_kepler10.py
"""

import lightkurve as lk
import matplotlib.pyplot as plt

TARGET = "Kepler-10"

def main():
    print(f"Searching for light curve data on {TARGET}...")
    search_result = lk.search_lightcurve(TARGET, mission="Kepler", cadence="long")
    print(search_result)

    if len(search_result) == 0:
        print("No results found. Check target name or mission.")
        return

    # Download the first quarter of data as a starting point
    lc = search_result[0].download()

    print(f"\nDownloaded light curve: {lc}")
    print(f"Time range: {lc.time.min()} to {lc.time.max()}")
    print(f"Number of data points: {len(lc.flux)}")

    # Basic plot — raw flux over time
    fig, ax = plt.subplots(figsize=(12, 4))
    lc.plot(ax=ax)
    ax.set_title(f"{TARGET} — Raw Light Curve (Quarter 1)")
    plt.tight_layout()
    plt.savefig("kepler10_raw_lightcurve.png", dpi=150)
    print("\nSaved plot to kepler10_raw_lightcurve.png")

    # Quick look: flatten (detrend) to remove stellar variability
    flat_lc = lc.flatten(window_length=401)
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    flat_lc.plot(ax=ax2)
    ax2.set_title(f"{TARGET} — Flattened/Detrended Light Curve")
    plt.tight_layout()
    plt.savefig("kepler10_flattened_lightcurve.png", dpi=150)
    print("Saved plot to kepler10_flattened_lightcurve.png")

if __name__ == "__main__":
    main()
