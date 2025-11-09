"""
gait_forceplate_realistic.py
--------------------------------
Bertec-style force-plate gait analysis.

- Reads CSV with header on line 5 (header=4)
- fs = 50 Hz
- Robust stance detection & unit/sign handling
- Plate 2 ONLY:
    * % stance / % swing (stride = start_i -> start_{i+1})
    * COP metrics over stance (AP/ML excursions, path length, mean speed)
- General summary (plates 1 & 2): peak/mean Fz, impulse, loading rate per contact
- Organized outputs:
    results/<participant>/<condition>/<trial>/
      <kind>__<participant>__<condition>__<trial>.csv/.png

COP convention (your lab, Bertec Right-Hand Rule):
    X = ML, Y = AP
You can flip AP/ML signs with COP_AP_FLIP / COP_ML_FLIP below if needed.
"""

from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ===== USER SETTINGS =====
participant_id  = "Blake"   # e.g., "Blake01"
trial_condition = "BL"      # e.g., "Shoes", "Barefoot", "Fast"
trial_id        = "T01"     # e.g., "BL_01", "Run_02"
file_path = "/Users/evanquan/Downloads/Blake/FullPilotBlake_BL/FullPilotBlake_BL_forces_2025_01_17_120728.csv"

fs = 50                 # Hz
filter_cutoff = 10      # Hz low-pass (good for 50 Hz force/COP at 50 Hz sampling)
threshold_N = 100       # N stance threshold (tune 80–150 N)
min_contact_s = 0.10    # s minimum stance duration
merge_gap_s = 0.06      # s merge micro-gaps between stance fragments
save_plots = True

# Plate 2 only scale: If your file is in kN, set to 1000.0; if already N, leave as 1.0
force_scale_plate2 = 1.0

# COP axis mapping (Bertec RHR in your lab): X = ML, Y = AP
# If you find positive AP points backward or positive ML points to the right,
# flip the signs here without touching the math below.
COP_AP_FLIP = False
COP_ML_FLIP = False

# Master output folder (relative to project)
output_root = Path("./results")


# ===== PATH HELPERS =====
def slug(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]", "", s)
    return s

p_slug = slug(participant_id)
c_slug = slug(trial_condition)
t_slug = slug(trial_id)

outdir = output_root / p_slug / c_slug / t_slug
outdir.mkdir(parents=True, exist_ok=True)

def outname(kind: str, ext: str) -> Path:
    """kind like 'plate2_contact_summary', ext='csv' or 'png'."""
    return outdir / f"{kind}__{p_slug}__{c_slug}__{t_slug}.{ext}"


# ===== PROCESSING HELPERS =====
def butter_lowpass_filter(data, cutoff, fs, order=4):
    if cutoff is None or cutoff <= 0:
        return data
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, data)

def find_fz_column(df, plate):
    """
    Prefer '{plate}:FZ'. If missing, fall back to '{plate}:FZR' + '{plate}:FZL'.
    Returns numpy array or None if not found.
    """
    col_main = f"{plate}:FZ"
    if col_main in df.columns:
        return df[col_main].astype(float).to_numpy()

    col_r = f"{plate}:FZR"
    col_l = f"{plate}:FZL"
    if col_r in df.columns and col_l in df.columns:
        return (df[col_r].astype(float) + df[col_l].astype(float)).to_numpy()

    return None

def normalize_units_and_sign(fz_raw, fs):
    """
    Use when data are already in N.
    - Make upward positive (Bertec often has upward negative),
    - Remove small DC offset using first ~0.5 s,
    - Clip extreme spikes for robustness.
    """
    fz = np.asarray(fz_raw, dtype=float)

    # Make upward positive
    if np.nanmedian(fz) < 0:
        fz = -fz
    fz = np.abs(fz)

    # Remove baseline (first ~0.5 s)
    baseline_frames = max(20, int(0.5 * fs))
    baseline = float(np.nanmedian(fz[:baseline_frames]))
    fz = fz - baseline
    fz[fz < 0] = 0.0

    # Clip extreme spikes (>6× a high percentile)
    bw_guess = np.nanpercentile(fz, 95)
    if np.isfinite(bw_guess) and bw_guess > 0:
        cap = 6.0 * bw_guess
        fz = np.clip(fz, 0, cap)

    return fz

def clean_vertical_force_with_scale(fz_raw, fs, scale=1.0):
    """
    Plate-2 cleaner that allows optional kN->N scaling via 'scale'.
    """
    fz = np.asarray(fz_raw, float) * float(scale)
    if np.nanmedian(fz) < 0:
        fz = -fz
    fz = np.abs(fz)
    baseline_frames = max(20, int(0.5 * fs))
    baseline = float(np.nanmedian(fz[:baseline_frames]))
    fz = fz - baseline
    fz[fz < 0] = 0.0
    return fz

def detect_contacts(force, fs, threshold=100, min_dur=0.10, merge_gap_s=0.06):
    """Returns [(start_idx, end_idx), ...] stance intervals."""
    above = force > threshold
    contacts, in_contact, start = [], False, None
    for i, val in enumerate(above):
        if val and not in_contact:
            start, in_contact = i, True
        elif not val and in_contact:
            end = i
            if (end - start) / fs >= min_dur:
                contacts.append((start, end))
            in_contact = False
    if in_contact and start is not None:
        contacts.append((start, len(force)))

    # merge micro-gaps
    if not contacts:
        return contacts
    merged = [contacts[0]]
    gap_frames = int(merge_gap_s * fs)
    for s, e in contacts[1:]:
        ps, pe = merged[-1]
        if s - pe <= gap_frames:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    return merged

def compute_metrics(fz, fs):
    """
    Basic stance metrics per contact (for general per-plate summary):
    contact time, peak/mean Fz, vertical impulse, max loading rate (first 150 ms).
    """
    contacts = detect_contacts(
        fz, fs,
        threshold=threshold_N,
        min_dur=min_contact_s,
        merge_gap_s=merge_gap_s
    )
    rows = []
    for idx, (s, e) in enumerate(contacts, start=1):
        seg = fz[s:e]
        if seg.size < 3:
            continue
        contact_time = (e - s) / fs
        peak_fz = float(np.max(seg))
        mean_fz = float(np.mean(seg))
        impulse = float(np.trapz(seg, dx=1/fs))

        # Max loading rate in first 150 ms of stance
        win = max(2, int(0.150 * fs))
        dFdt = np.gradient(seg) * fs
        lr = float(np.max(dFdt[:win])) if dFdt.size > 1 else np.nan

        rows.append({
            "contact_index": idx,
            "start_time_s": s / fs,
            "end_time_s": e / fs,
            "contact_time_s": contact_time,
            "peak_Fz_N": peak_fz,
            "mean_Fz_N": mean_fz,
            "vertical_impulse_Ns": impulse,
            "max_loading_rate_Ns": lr
        })
    return pd.DataFrame(rows)

def get_cop_plate2(df, fz):
    """
    Returns (copx, copy) in meters for Plate 2.
    Uses 2:COPX/2:COPY if present (auto mm->m),
    else computes from 2:MX/2:MY and Fz (auto N·mm->N·m).
    """
    if ("2:COPX" in df.columns) and ("2:COPY" in df.columns):
        copx = df["2:COPX"].astype(float).to_numpy()
        copy = df["2:COPY"].astype(float).to_numpy()
        # If looks like mm, convert to m
        if np.nanmedian(np.abs(copx)) > 1.0 or np.nanmedian(np.abs(copy)) > 1.0:
            copx, copy = copx / 1000.0, copy / 1000.0
        return copx, copy

    if ("2:MX" in df.columns) and ("2:MY" in df.columns):
        Mx = df["2:MX"].astype(float).to_numpy()
        My = df["2:MY"].astype(float).to_numpy()
        # If N·mm, convert to N·m
        if np.nanmedian(np.abs(Mx)) > 100.0 or np.nanmedian(np.abs(My)) > 100.0:
            Mx *= 0.001
            My *= 0.001
        eps = 1e-6
        Fz_safe = np.where(np.abs(fz) < eps, np.nan, fz)
        copx = (-My) / Fz_safe
        copy = ( Mx) / Fz_safe
        return copx, copy

    n = len(fz)
    return np.full(n, np.nan), np.full(n, np.nan)

def cop_metrics(cop1_seg, cop2_seg, fs):
    """
    Generic COP metrics on two series (interpreted as AP/ML later):
    Returns (axis1_excursion_m, axis2_excursion_m, path_length_m, mean_speed_mps).
    """
    x = np.asarray(cop1_seg, float); y = np.asarray(cop2_seg, float)
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2:
        return np.nan, np.nan, np.nan, np.nan
    x, y = x[mask], y[mask]
    x_exc = float(np.nanmax(x) - np.nanmin(x))
    y_exc = float(np.nanmax(y) - np.nanmin(y))
    dx, dy = np.diff(x), np.diff(y)
    path = float(np.sum(np.sqrt(dx*dx + dy*dy)))
    mean_speed = float(path / (len(x) / fs))
    return x_exc, y_exc, path, mean_speed


# ===== MAIN =====
print(f"📂 Writing outputs to: {outdir.resolve()}")
print("🔍 Loading file with header=4...")
df = pd.read_csv(file_path, header=4)  # header row is line 5
print(f"✅ Columns ({len(df.columns)}):", list(df.columns)[:12], "...")

# ---------- Plate 2 ONLY: % stance / % swing + COP metrics ----------
# Plate 2 vertical force
fz2_raw = find_fz_column(df, plate=2)
if fz2_raw is None:
    raise RuntimeError("Plate 2 Fz not found (need 2:FZ or 2:FZR+2:FZL).")

# Clean + filter (explicit scale set above)
fz2 = clean_vertical_force_with_scale(fz2_raw, fs, scale=force_scale_plate2)
fz2_f = butter_lowpass_filter(fz2, filter_cutoff, fs)

# Detect stances (Plate 2)
contacts2 = detect_contacts(
    fz2_f, fs,
    threshold=threshold_N,
    min_dur=min_contact_s,
    merge_gap_s=merge_gap_s
)
print(f"→ Plate 2 contacts detected: {len(contacts2)}")

# Prepare COP time series for Plate 2 (uses filtered Fz for safety)
# Bertec RHR in your lab: X = ML, Y = AP
copx2, copy2 = get_cop_plate2(df, fz2_f)   # raw plate axes
cop_ml2 = copx2.copy()  # X -> ML
cop_ap2 = copy2.copy()  # Y -> AP

# Optional sign flips (lab preference)
if COP_AP_FLIP:
    cop_ap2 = -cop_ap2
if COP_ML_FLIP:
    cop_ml2 = -cop_ml2

# % stance / % swing per stride (Plate 2 only)
rows_p2 = []
for i, (s, e) in enumerate(contacts2, start=1):
    contact_time = (e - s) / fs

    # stride timing: start_i -> start_{i+1}
    if i < len(contacts2):
        stride_time = (contacts2[i][0] - s) / fs
        stance_pct = 100.0 * contact_time / stride_time if stride_time > 0 else np.nan
        swing_pct  = 100.0 - stance_pct if np.isfinite(stance_pct) else np.nan
    else:
        stride_time, stance_pct, swing_pct = np.nan, np.nan, np.nan

    # Basic Fz metrics
    seg = fz2_f[s:e]
    peak_fz = float(np.max(seg)) if seg.size else np.nan
    mean_fz = float(np.mean(seg)) if seg.size else np.nan
    impulse = float(np.trapz(seg, dx=1/fs)) if seg.size else np.nan

    # COP metrics over stance (AP/ML)
    ap_seg = cop_ap2[s:e] if cop_ap2 is not None else None
    ml_seg = cop_ml2[s:e] if cop_ml2 is not None else None
    ap_exc, ml_exc, path_len, mean_speed = cop_metrics(ap_seg, ml_seg, fs)

    rows_p2.append({
        "participant": participant_id,
        "condition": trial_condition,
        "trial": trial_id,
        "plate": 2,
        "contact_index": i,
        "start_time_s": s / fs,
        "end_time_s": e / fs,
        "contact_time_s": contact_time,
        "stride_time_s": stride_time,
        "stance_pct_of_stride": stance_pct,
        "swing_pct_of_stride": swing_pct,
        "peak_Fz_N": peak_fz,
        "mean_Fz_N": mean_fz,
        "vertical_impulse_Ns": impulse,
        "cop_AP_excursion_m": ap_exc,
        "cop_ML_excursion_m": ml_exc,
        "cop_path_length_m": path_len,
        "cop_mean_speed_mps": mean_speed,
    })

plate2_out = pd.DataFrame(rows_p2)
plate2_csv = outname("plate2_contact_summary", "csv")
plate2_out.to_csv(plate2_csv, index=False)
print(f"💾 Saved → {plate2_csv}")

# Optional Plate 2 force plot (detections)
if save_plots:
    plt.figure(figsize=(10, 4))
    plt.plot(fz2_f, label="Plate 2 Fz (filtered)")
    plt.axhline(threshold_N, linestyle="--", label="Threshold")
    for s, e in contacts2:
        plt.axvspan(s, e, alpha=0.2)
    plt.xlabel("Frame"); plt.ylabel("Vertical Force (N)")
    plt.title("Plate 2 Vertical GRF with detected stances")
    plt.legend(); plt.tight_layout()
    plt.savefig(outname("plate2_Fz_plot", "png"), dpi=150); plt.close()
    print(f"🖼  Saved → {outname('plate2_Fz_plot','png')}")

# ---- COP plots (Plate 2, AP/ML) ----
# 1) Trajectory AP vs ML with stance highlighting
mask = ~(np.isnan(cop_ap2) | np.isnan(cop_ml2))
cop_ap2_plot = cop_ap2[mask]; cop_ml2_plot = cop_ml2[mask]
plt.figure(figsize=(6, 6))
plt.plot(cop_ap2_plot, cop_ml2_plot, linewidth=1.0)
for s, e in contacts2:
    plt.plot(cop_ap2[s:e], cop_ml2[s:e], linewidth=2.0)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("COP AP (m)")
plt.ylabel("COP ML (m)")
plt.title("Plate 2 COP Trajectory (AP vs ML)")
plt.grid(True); plt.tight_layout()
plt.savefig(outname("plate2_COP_trajectory_APML", "png"), dpi=150); plt.close()
print(f"🖼  Saved → {outname('plate2_COP_trajectory_APML','png')}")

# ---------- General per-plate summary (plates 1 & 2), optional keep ----------
plates = [1, 2]
all_rows = []
for plate in plates:
    fz_raw = find_fz_column(df, plate)
    if fz_raw is None:
        print(f"⚠️ Could not find {plate}:FZ or {plate}:FZR+{plate}:FZL — skipping plate {plate}")
        continue

    fz = normalize_units_and_sign(fz_raw, fs)
    fz_f = butter_lowpass_filter(fz, filter_cutoff, fs)

    res = compute_metrics(fz_f, fs)
    if res.empty:
        print(f"⚠️ No stance detected for plate {plate} with threshold {threshold_N} N.")
        continue
    res["plate"] = plate
    res["participant"] = participant_id
    res["condition"] = trial_condition
    res["trial"] = trial_id
    all_rows.append(res)

    if save_plots:
        plt.figure(figsize=(10, 4))
        plt.plot(fz_f)
        plt.axhline(threshold_N, linestyle="--")
        plt.xlabel("Frame")
        plt.ylabel("Vertical Force (N)")
        plt.title(f"Vertical GRF (filtered) – Plate {plate}")
        plt.tight_layout()
        plt.savefig(outname(f"plate{plate}_Fz_plot", "png"), dpi=150)
        plt.close()

if all_rows:
    out = pd.concat(all_rows, ignore_index=True)
    summary_csv = outname("gait_forceplate_summary", "csv")
    out.to_csv(summary_csv, index=False)
    print(f"💾 Saved → {summary_csv}")
    if save_plots:
        print(f"🖼  Saved → {outname('plate1_Fz_plot','png')} / {outname('plate2_Fz_plot','png')}")
else:
    print("ℹ️ Skipped general summary (no valid results).")
