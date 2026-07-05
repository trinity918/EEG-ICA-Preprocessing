"""
SCIENTIFIC ANALYSIS - N400 ERP (read-only; uses ONLY frozen preprocessing outputs)
==================================================================================
Reads the frozen sub-XX_desc-N400_epo.fif (AutoReject-cleaned, 30 Hz, avg-ref, metadata-
tagged). No preprocessing is rerun or modified.

SCOPE / STATISTICAL HONESTY
  * Usable subjects: Pilot_1, Pilot_2 (Pilot_3 epoching blocked by a trigger mismatch).
  * N = 2 -> GROUP-LEVEL inference is NOT warranted. This stage is per-subject and
    descriptive, with within-subject (trial-level) cluster statistics as EXPLORATORY
    evidence, plus a 2-subject descriptive grand average. No group significance is claimed.

Design: visual rhyme judgement; ERP time-locked to exp_stim2 (2nd word).
  Primary contrast : NonRhyme vs Rhyme (classic N400: NonRhyme more negative, ~300-500 ms,
                     central-parietal).
  Secondary        : 2x2 Orthography x Phonology (O+P+, O+P-, O-P+, O-P-) - descriptive.

Metrics (defined):
  * N400 mean amplitude = mean voltage over 300-500 ms, averaged across ROI {Cz,Pz,CP1,CP2}.
  * aSME (analytic Standardized Measurement Error, Luck 2021) = SD across single-trial
    N400 mean-amplitudes / sqrt(n_trials).
  * Within-subject test = spatiotemporal cluster-based permutation F-test across trials
    (independent conditions), post-stimulus 0-0.8 s, cluster-forming p<0.05, 1000 perms.
"""
import warnings, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from pathlib import Path
import mne
from mne.stats import spatio_temporal_cluster_test
mne.set_log_level("ERROR")

OUT = Path("derivatives/analysis/n400"); (OUT / "figures").mkdir(parents=True, exist_ok=True)
SUBJECTS = ["01-Pilot", "02"]
ROI = ["Cz", "Pz", "CP1", "CP2"]
N400 = (0.30, 0.50)
OP = ["O+P+", "O+P-", "O-P+", "O-P-"]

def savef(f, name):
    p = OUT / "figures" / f"{name}.png"; f.savefig(p, dpi=140, bbox_inches="tight"); plt.close(f); return str(p)

def load(subject):
    ep = mne.read_epochs(f"derivatives/sub-{subject}/eeg/sub-{subject}_desc-N400_epo.fif",
                         preload=True, verbose=False)
    ep.metadata["_rhyme"] = ep.metadata["exprhyme_cond"].str.lower().str.strip()
    ep.metadata["_op"] = ep.metadata["expcondition"].str.upper().str.strip()
    return ep

def roi_trace(ep, roi):
    """(n_trials, n_times) ROI-mean single-trial traces in uV."""
    return ep.get_data(picks=roi).mean(axis=1) * 1e6

def n400_amp(ep, roi):
    """Per-trial N400 mean amplitude (uV) over 300-500 ms, ROI mean."""
    tmask = (ep.times >= N400[0]) & (ep.times <= N400[1])
    return roi_trace(ep, roi)[:, tmask].mean(axis=1)

def asme(amps):
    return float(np.std(amps, ddof=1) / np.sqrt(len(amps)))

results = {"scope": "per-subject descriptive; N=2 usable; no group inference", "subjects": {}}
rows = []; ga_store = {"rhyme": [], "nonrhyme": []}

for subj in SUBJECTS:
    ep = load(subj)
    r = ep[ep.metadata["_rhyme"] == "rhyme"]
    n = ep[ep.metadata["_rhyme"] == "nonrhyme"]
    ev_r, ev_n = r.average(), n.average()
    ga_store["rhyme"].append(ev_r); ga_store["nonrhyme"].append(ev_n)
    t = ep.times

    # --- N400 amplitudes + aSME (ROI, 300-500 ms) ---
    a_r, a_n = n400_amp(r, ROI), n400_amp(n, ROI)
    diff_amp = float(a_n.mean() - a_r.mean())            # NonRhyme - Rhyme
    # exploratory independent-samples t across trials (descriptive only)
    from scipy import stats as sstats
    tval, pval = sstats.ttest_ind(a_n, a_r, equal_var=False)

    # --- Fig 1: rhyme vs nonrhyme ROI mean +/- SEM + difference wave ---
    pr, pn = roi_trace(r, ROI), roi_trace(n, ROI)
    mr, sr = pr.mean(0), pr.std(0)/np.sqrt(pr.shape[0])
    mn, sn = pn.mean(0), pn.std(0)/np.sqrt(pn.shape[0])
    dw = mn - mr; sd = np.sqrt(sr**2 + sn**2)
    f, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for m, s, lb, c in [(mr, sr, f"Rhyme (n={pr.shape[0]})", "tab:blue"),
                        (mn, sn, f"NonRhyme (n={pn.shape[0]})", "tab:red")]:
        ax[0].plot(t, m, c, lw=1.4, label=lb); ax[0].fill_between(t, m-s, m+s, color=c, alpha=0.25)
    ax[0].axvspan(*N400, color="gray", alpha=0.12); ax[0].axvline(0, color="k", lw=.5); ax[0].axhline(0, color="k", lw=.5)
    ax[0].invert_yaxis(); ax[0].legend(fontsize=9)
    ax[0].set(ylabel="uV (neg up)", title=f"[{subj}] N400: Rhyme vs NonRhyme, ROI {ROI} mean +/- SEM")
    ax[1].plot(t, dw, "k", lw=1.4, label="NonRhyme - Rhyme"); ax[1].fill_between(t, dw-sd, dw+sd, color="0.5", alpha=0.3)
    ax[1].axvspan(*N400, color="gray", alpha=0.12); ax[1].axvline(0, color="k", lw=.5); ax[1].axhline(0, color="k", lw=.5)
    ax[1].invert_yaxis(); ax[1].legend(fontsize=9)
    ax[1].set(xlabel="Time (s)", ylabel="uV (neg up)",
              title=f"Difference wave (N400 window mean diff = {diff_amp:+.2f} uV, exploratory t={tval:.2f} p={pval:.3f})")
    savef(f, f"sub-{subj}_rhyme_vs_nonrhyme")

    # --- Fig 2: scalp topographies (N400 window) ---
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    tmask = (t >= N400[0]) & (t <= N400[1])
    for k, (evk, ttl) in enumerate([(ev_r, "Rhyme"), (ev_n, "NonRhyme")]):
        d = evk.get_data()[:, tmask].mean(1)
        mne.viz.plot_topomap(d, evk.info, axes=ax[k], show=False, contours=4)
        ax[k].set_title(f"{ttl} 300-500 ms")
    dwt = ev_n.get_data()[:, tmask].mean(1) - ev_r.get_data()[:, tmask].mean(1)
    im, _ = mne.viz.plot_topomap(dwt, ev_r.info, axes=ax[2], show=False, contours=4)
    ax[2].set_title("NonRhyme - Rhyme")
    fig.colorbar(im, ax=ax[2], shrink=0.7); fig.suptitle(f"[{subj}] N400-window scalp distribution (V)")
    savef(fig, f"sub-{subj}_topography")

    # --- Fig 3: 2x2 O/P ROI waveforms (descriptive) ---
    fig, ax = plt.subplots(figsize=(9, 5)); opamp = {}
    colmap = {"O+P+": "navy", "O+P-": "darkred", "O-P+": "tab:cyan", "O-P-": "salmon"}
    for cond in OP:
        sub = ep[ep.metadata["_op"] == cond]
        if len(sub) == 0: continue
        m = roi_trace(sub, ROI).mean(0); ax.plot(t, m, color=colmap[cond], lw=1.3, label=f"{cond} (n={len(sub)})")
        opamp[cond] = float(n400_amp(sub, ROI).mean())
    ax.axvspan(*N400, color="gray", alpha=0.12); ax.axvline(0, color="k", lw=.5); ax.axhline(0, color="k", lw=.5)
    ax.invert_yaxis(); ax.legend(fontsize=8)
    ax.set(xlabel="Time (s)", ylabel="uV (neg up)", title=f"[{subj}] 2x2 Orthography x Phonology, ROI mean (descriptive)")
    savef(fig, f"sub-{subj}_2x2_OP")

    # --- within-subject spatiotemporal cluster permutation (exploratory) ---
    adj, _ = mne.channels.find_ch_adjacency(ep.info, "eeg")
    post = ep.times >= 0
    Xr = r.get_data()[:, :, post].transpose(0, 2, 1)   # (trials, times, chans)
    Xn = n.get_data()[:, :, post].transpose(0, 2, 1)
    Fobs, clusters, cluster_pv, _ = spatio_temporal_cluster_test(
        [Xn, Xr], n_permutations=1000, adjacency=adj, n_jobs=1, seed=42, verbose=False)
    sig = [float(p) for p in cluster_pv if p < 0.05]
    min_p = float(cluster_pv.min()) if len(cluster_pv) else None

    # main effects (descriptive)
    def op_mean(pred):
        idx = ep.metadata["_op"].isin(pred)
        return float(n400_amp(ep[np.where(idx.values)[0]], ROI).mean())
    O_main = op_mean(["O+P+", "O+P-"]) - op_mean(["O-P+", "O-P-"])   # O+ minus O-
    P_main = op_mean(["O+P+", "O-P+"]) - op_mean(["O+P-", "O-P-"])   # P+ minus P-

    results["subjects"][subj] = dict(
        n_trials=len(ep), n_rhyme=len(r), n_nonrhyme=len(n),
        n400_amp_rhyme=float(a_r.mean()), n400_amp_nonrhyme=float(a_n.mean()),
        n400_diff_nonrhyme_minus_rhyme=diff_amp,
        aSME_rhyme=asme(a_r), aSME_nonrhyme=asme(a_n),
        exploratory_ttest_t=float(tval), exploratory_ttest_p=float(pval),
        op_n400_amp=opamp, orthography_main_effect=O_main, phonology_main_effect=P_main,
        cluster_min_p=min_p, n_sig_clusters=len(sig))
    rows.append(dict(subject=subj, n_trials=len(ep),
                     N400_rhyme_uV=round(float(a_r.mean()), 2), N400_nonrhyme_uV=round(float(a_n.mean()), 2),
                     diff_NR_minus_R_uV=round(diff_amp, 2), aSME_rhyme=round(asme(a_r), 2),
                     aSME_nonrhyme=round(asme(a_n), 2), exploratory_t=round(float(tval), 2),
                     exploratory_p=round(float(pval), 3), cluster_min_p=(round(min_p, 3) if min_p else None),
                     n_sig_clusters=len(sig)))
    print(f"[{subj}] N400 R={a_r.mean():.2f} NR={a_n.mean():.2f} diff={diff_amp:+.2f} uV | "
          f"aSME R/NR={asme(a_r):.2f}/{asme(a_n):.2f} | t={tval:.2f} p={pval:.3f} | "
          f"cluster min p={min_p} sig={len(sig)}")

# --- Grand average (2 subjects, DESCRIPTIVE, underpowered) ---
ga_r = mne.grand_average(ga_store["rhyme"]); ga_n = mne.grand_average(ga_store["nonrhyme"])
f, ax = plt.subplots(figsize=(9, 5)); tt = ga_r.times
mr = ga_r.get_data(picks=ROI).mean(0)*1e6; mn = ga_n.get_data(picks=ROI).mean(0)*1e6
ax.plot(tt, mr, "tab:blue", lw=1.5, label="Rhyme (GA)"); ax.plot(tt, mn, "tab:red", lw=1.5, label="NonRhyme (GA)")
ax.plot(tt, mn-mr, "k", lw=1.2, ls="--", label="NonRhyme - Rhyme")
ax.axvspan(*N400, color="gray", alpha=0.12); ax.axvline(0, color="k", lw=.5); ax.axhline(0, color="k", lw=.5)
ax.invert_yaxis(); ax.legend(fontsize=9)
ax.set(xlabel="Time (s)", ylabel="uV (neg up)",
       title="Grand average (N=2, DESCRIPTIVE ONLY - not group inference): Rhyme vs NonRhyme, ROI mean")
savef(f, "grand_average_N2_descriptive")

df = pd.DataFrame(rows); df.to_csv(OUT / "n400_results.csv", index=False)
json.dump(results, open(OUT / "n400_results.json", "w"), indent=2, default=str)
print("\n=== N400 ANALYSIS (per-subject, descriptive) ===")
print(df.to_string(index=False))
print(f"\nfigures + tables -> {OUT}")
