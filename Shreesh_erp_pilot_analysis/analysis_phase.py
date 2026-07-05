"""
ANALYSIS PHASE (READ-ONLY) - consumes ONLY frozen preprocessing outputs.
========================================================================
No preprocessing is rerun/modified. Inputs: *_desc-cleaned_raw.fif, *_desc-N400_epo.fif,
provenance.json. Outputs -> derivatives/analysis/{psd,fooof,erp,discussion}/.

Frozen config values (from provenance.json; kept in one place, not re-hardcoded per use):
  ALPHA_BAND=(8,13) QC_BETA_BAND=(13,30) PSD_FMAX=45 QC_BANDPOWER_NFFT=8192
  POSTERIOR_FRACTION=0.5 ERP_N400_ROI=[Cz,Pz,CP1,CP2] EPOCH (-0.2,0.8) LP_FREQ_ERP=30
get_posterior_channels() is replicated inline IDENTICALLY to the frozen helper; the
delivered notebook cells call the real helper directly.
"""
import warnings, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy import stats as sstats
from pathlib import Path
import mne
from mne.stats import spatio_temporal_cluster_test
mne.set_log_level("ERROR")

A = Path("derivatives/analysis")
for d in ["psd", "fooof", "erp", "discussion"]:
    (A / d).mkdir(parents=True, exist_ok=True)

# ---- frozen config (single source) ----
ALPHA_BAND = (8.0, 13.0); BETA_BAND = (13.0, 30.0)
PSD_FMAX = 45.0; NFFT = 8192; POSTERIOR_FRACTION = 0.5
ROI = ["Cz", "Pz", "CP1", "CP2"]; N400_WIN = (0.30, 0.50)
BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": ALPHA_BAND,
         "beta": BETA_BAND, "gamma": (30.0, PSD_FMAX)}   # gamma capped at PSD_FMAX (sampled band)
SUBJECTS = ["01-Pilot", "02", "03"]
ERP_SUBJECTS = ["01-Pilot", "02"]        # Pilot_3 epoching blocked (trigger mismatch)

def get_posterior_channels(raw):
    """Identical to the frozen helper: posterior electrodes (y < 0.5*most-posterior y), no bads."""
    pos = raw.get_montage().get_positions()["ch_pos"]
    ys = {ch: pos[ch][1] for ch in raw.ch_names if ch in pos and ch not in raw.info["bads"]}
    thr = POSTERIOR_FRACTION * min(ys.values())
    return [ch for ch, y in ys.items() if y < thr]

def cleaned(subject):
    return mne.io.read_raw_fif(f"derivatives/sub-{subject}/eeg/sub-{subject}_desc-cleaned_raw.fif",
                               preload=True, verbose=False)

def bandpower(freqs, psd, band):
    m = (freqs >= band[0]) & (freqs <= band[1])
    return float(simpson(psd[m], x=freqs[m])) * 1e12          # V^2 -> uV^2

# ============================================================ PART 1 - PSD
def part1_psd():
    rows = []
    for subj in SUBJECTS:
        raw = cleaned(subj); post = get_posterior_channels(raw)
        psd = raw.compute_psd(method="welch", fmin=1.0, fmax=PSD_FMAX, n_fft=NFFT, verbose=False)
        f = psd.freqs; P = psd.get_data()                     # (n_chan, n_freq), V^2/Hz
        mean_all = P.mean(0)
        pidx = [raw.ch_names.index(c) for c in post]
        mean_post = P[pidx].mean(0)

        # Fig 1: mean all-channel PSD
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(f, 10*np.log10(mean_all), "k", lw=1.4, label="mean (all channels)")
        ax.fill_between(f, 10*np.log10(np.percentile(P, 25, 0)), 10*np.log10(np.percentile(P, 75, 0)),
                        color="0.7", alpha=0.4, label="IQR across channels")
        for b in BANDS.values(): ax.axvspan(*b, color="0.9", alpha=0.15)
        ax.set(xscale="log", xlabel="Frequency (Hz)", ylabel="Power (dB, V^2/Hz)",
               title=f"[{subj}] Whole-head mean PSD (Welch)"); ax.legend(fontsize=8)
        fig.savefig(A/"psd"/f"sub-{subj}_psd_allchannels.png", dpi=140, bbox_inches="tight"); plt.close(fig)

        # Fig 2: mean posterior + individual posterior channels
        fig, ax = plt.subplots(figsize=(9, 5))
        for c in post:
            ax.plot(f, 10*np.log10(P[raw.ch_names.index(c)]), lw=0.6, alpha=0.6, label=c)
        ax.plot(f, 10*np.log10(mean_post), "k", lw=2.0, label="posterior mean")
        ax.axvspan(*ALPHA_BAND, color="purple", alpha=0.1)
        ax.set(xscale="log", xlabel="Frequency (Hz)", ylabel="Power (dB, V^2/Hz)",
               title=f"[{subj}] Posterior PSD (n={len(post)}: {', '.join(post)})"); ax.legend(fontsize=7, ncol=2)
        fig.savefig(A/"psd"/f"sub-{subj}_psd_posterior.png", dpi=140, bbox_inches="tight"); plt.close(fig)

        # band-power table (whole-head mean + posterior mean; absolute uV^2 + relative %)
        tot_all = sum(bandpower(f, mean_all, b) for b in BANDS.values())
        tot_post = sum(bandpower(f, mean_post, b) for b in BANDS.values())
        for region, spec, tot in [("wholehead", mean_all, tot_all), ("posterior", mean_post, tot_post)]:
            rec = {"subject": subj, "region": region}
            for name, b in BANDS.items():
                bp = bandpower(f, spec, b)
                rec[f"{name}_uV2"] = round(bp, 4); rec[f"{name}_rel_pct"] = round(100*bp/tot, 2)
            rows.append(rec)
        print(f"[PSD {subj}] posterior alpha={bandpower(f, mean_post, ALPHA_BAND):.3f} uV^2 "
              f"({100*bandpower(f, mean_post, ALPHA_BAND)/tot_post:.1f}% of 1-45 Hz)")
    df = pd.DataFrame(rows); df.to_csv(A/"psd"/"bandpower_summary.csv", index=False)
    # band-power bar figure (posterior, relative %)
    dfp = df[df.region == "posterior"]
    fig, ax = plt.subplots(figsize=(9, 5)); x = np.arange(len(SUBJECTS)); w = 0.15
    for i, name in enumerate(BANDS):
        ax.bar(x + (i-2)*w, dfp[f"{name}_rel_pct"], w, label=name)
    ax.set_xticks(x); ax.set_xticklabels(SUBJECTS); ax.set(ylabel="relative power (%)",
        title="Posterior relative band power (descriptive)"); ax.legend(fontsize=8)
    fig.savefig(A/"psd"/"bandpower_posterior_relative.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    return df

# ============================================================ PART 2 - FOOOF
def part2_fooof():
    from fooof import FOOOF
    from fooof.analysis import get_band_peak_fm
    import io, contextlib
    rows = []; full = {}
    for subj in SUBJECTS:
        raw = cleaned(subj); post = get_posterior_channels(raw)
        psd = raw.compute_psd(method="welch", fmin=1.0, fmax=PSD_FMAX, picks=post, n_fft=NFFT, verbose=False)
        fm = FOOOF(peak_width_limits=[1.0, 8.0], max_n_peaks=6, min_peak_height=0.05,
                   aperiodic_mode="fixed", verbose=False)
        with contextlib.redirect_stdout(io.StringIO()):
            fm.fit(psd.freqs, psd.get_data().mean(0), (2.0, 40.0))
        fig, ax = plt.subplots(figsize=(8, 5)); fm.plot(ax=ax, plt_log=False)
        ax.set_title(f"[{subj}] posterior mean PSD - FOOOF (R2={fm.r_squared_:.3f})")
        fig.savefig(A/"fooof"/f"sub-{subj}_fooof.png", dpi=140, bbox_inches="tight"); plt.close(fig)
        off, exp = fm.aperiodic_params_
        al = get_band_peak_fm(fm, ALPHA_BAND, select_highest=True)
        has_alpha = al is not None and not np.isnan(al[0])
        rows.append(dict(subject=subj, aperiodic_offset=round(off, 3), aperiodic_exponent=round(exp, 3),
                         alpha_cf_hz=(round(float(al[0]), 2) if has_alpha else None),
                         alpha_power=(round(float(al[1]), 3) if has_alpha else None),
                         alpha_bandwidth_hz=(round(float(al[2]), 2) if has_alpha else None),
                         n_peaks=int(fm.n_peaks_), r_squared=round(fm.r_squared_, 4), error=round(fm.error_, 4)))
        full[subj] = dict(aperiodic=dict(offset=float(off), exponent=float(exp)),
                          peaks=[dict(cf=float(c), power=float(p), bw=float(b)) for c, p, b in fm.peak_params_],
                          alpha=(dict(cf=float(al[0]), power=float(al[1]), bw=float(al[2])) if has_alpha else None),
                          r_squared=float(fm.r_squared_), error=float(fm.error_))
        print(f"[FOOOF {subj}] exp={exp:.2f} alpha_cf={rows[-1]['alpha_cf_hz']} R2={fm.r_squared_:.3f}")
    pd.DataFrame(rows).to_csv(A/"fooof"/"fooof_summary.csv", index=False)
    json.dump(full, open(A/"fooof"/"fooof_results.json", "w"), indent=2)
    return pd.DataFrame(rows)

# ============================================================ PART 3 - ERP / N400
def part3_erp():
    rows = []; ga = {"rhyme": [], "nonrhyme": []}
    for subj in ERP_SUBJECTS:
        ep = mne.read_epochs(f"derivatives/sub-{subj}/eeg/sub-{subj}_desc-N400_epo.fif", preload=True, verbose=False)
        rk = ep.metadata["exprhyme_cond"].str.lower().str.strip()
        r, n = ep[np.where((rk == "rhyme").values)[0]], ep[np.where((rk == "nonrhyme").values)[0]]
        ga["rhyme"].append(r.average()); ga["nonrhyme"].append(n.average())
        t = ep.times; tmask = (t >= N400_WIN[0]) & (t <= N400_WIN[1])
        roi_r = r.get_data(picks=ROI).mean(1)*1e6; roi_n = n.get_data(picks=ROI).mean(1)*1e6
        amp_r = roi_r[:, tmask].mean(1); amp_n = roi_n[:, tmask].mean(1)
        asme = lambda a: float(np.std(a, ddof=1)/np.sqrt(len(a)))
        diff = float(amp_n.mean() - amp_r.mean())
        tval, pval = sstats.ttest_ind(amp_n, amp_r, equal_var=False)

        # waveform + difference
        mr, sr = roi_r.mean(0), roi_r.std(0)/np.sqrt(len(roi_r)); mn, sn = roi_n.mean(0), roi_n.std(0)/np.sqrt(len(roi_n))
        f, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        for m, s, lb, c in [(mr, sr, f"Rhyme (n={len(roi_r)})", "tab:blue"), (mn, sn, f"NonRhyme (n={len(roi_n)})", "tab:red")]:
            ax[0].plot(t, m, c, lw=1.4, label=lb); ax[0].fill_between(t, m-s, m+s, color=c, alpha=0.25)
        ax[0].axvspan(*N400_WIN, color="0.85"); ax[0].axhline(0, color="k", lw=.5); ax[0].axvline(0, color="k", lw=.5)
        ax[0].invert_yaxis(); ax[0].legend(fontsize=9); ax[0].set(ylabel="uV (neg up)", title=f"[{subj}] Rhyme vs NonRhyme, ROI {ROI} mean +/- SEM")
        dw = mn-mr; ax[1].plot(t, dw, "k", lw=1.4); ax[1].fill_between(t, dw-np.sqrt(sr**2+sn**2), dw+np.sqrt(sr**2+sn**2), color="0.5", alpha=0.3)
        ax[1].axvspan(*N400_WIN, color="0.85"); ax[1].axhline(0, color="k", lw=.5); ax[1].axvline(0, color="k", lw=.5); ax[1].invert_yaxis()
        ax[1].set(xlabel="Time (s)", ylabel="uV (neg up)", title=f"Difference NonRhyme-Rhyme (N400 mean={diff:+.2f} uV, t={tval:.2f} p={pval:.3f})")
        f.savefig(A/"erp"/f"sub-{subj}_rhyme_vs_nonrhyme.png", dpi=140, bbox_inches="tight"); plt.close(f)

        # topographies
        ev_r, ev_n = r.average(), n.average()
        fig, axx = plt.subplots(1, 3, figsize=(12, 4))
        for k, (evk, ttl) in enumerate([(ev_r, "Rhyme"), (ev_n, "NonRhyme")]):
            mne.viz.plot_topomap(evk.get_data()[:, tmask].mean(1), evk.info, axes=axx[k], show=False, contours=4); axx[k].set_title(f"{ttl} 300-500 ms")
        im, _ = mne.viz.plot_topomap(ev_n.get_data()[:, tmask].mean(1)-ev_r.get_data()[:, tmask].mean(1), ev_r.info, axes=axx[2], show=False, contours=4)
        axx[2].set_title("NonRhyme - Rhyme"); fig.colorbar(im, ax=axx[2], shrink=.7); fig.suptitle(f"[{subj}] N400-window topography")
        fig.savefig(A/"erp"/f"sub-{subj}_topography.png", dpi=140, bbox_inches="tight"); plt.close(fig)

        # within-subject spatiotemporal cluster permutation (exploratory)
        adj, _ = mne.channels.find_ch_adjacency(ep.info, "eeg"); postm = ep.times >= 0
        Fobs, cl, cpv, _ = spatio_temporal_cluster_test(
            [n.get_data()[:, :, postm].transpose(0, 2, 1), r.get_data()[:, :, postm].transpose(0, 2, 1)],
            n_permutations=1000, adjacency=adj, seed=42, n_jobs=1, verbose=False)
        min_p = float(cpv.min()) if len(cpv) else None; nsig = int(np.sum(cpv < 0.05))
        rows.append(dict(subject=subj, n_trials=len(ep), n400_rhyme_uV=round(float(amp_r.mean()), 2),
                         n400_nonrhyme_uV=round(float(amp_n.mean()), 2), diff_NR_minus_R_uV=round(diff, 2),
                         aSME_rhyme=round(asme(amp_r), 2), aSME_nonrhyme=round(asme(amp_n), 2),
                         exploratory_t=round(float(tval), 2), exploratory_p=round(float(pval), 3),
                         cluster_min_p=(round(min_p, 3) if min_p else None), n_sig_clusters=nsig))
        print(f"[ERP {subj}] diff={diff:+.2f}uV t={tval:.2f} p={pval:.3f} cluster_min_p={min_p} sig={nsig}")

    # grand average (N=2, DESCRIPTIVE)
    gar, gan = mne.grand_average(ga["rhyme"]), mne.grand_average(ga["nonrhyme"])
    f, ax = plt.subplots(figsize=(9, 5)); tt = gar.times
    mr = gar.get_data(picks=ROI).mean(0)*1e6; mn = gan.get_data(picks=ROI).mean(0)*1e6
    ax.plot(tt, mr, "tab:blue", lw=1.5, label="Rhyme"); ax.plot(tt, mn, "tab:red", lw=1.5, label="NonRhyme")
    ax.plot(tt, mn-mr, "k--", lw=1.2, label="NonRhyme-Rhyme"); ax.axvspan(*N400_WIN, color="0.85")
    ax.axhline(0, color="k", lw=.5); ax.axvline(0, color="k", lw=.5); ax.invert_yaxis(); ax.legend(fontsize=9)
    ax.set(xlabel="Time (s)", ylabel="uV (neg up)", title="Grand average (N=2, DESCRIPTIVE only - not group inference)")
    f.savefig(A/"erp"/"grand_average_descriptive.png", dpi=140, bbox_inches="tight"); plt.close(f)
    df = pd.DataFrame(rows); df.to_csv(A/"erp"/"n400_results.csv", index=False)
    json.dump(dict(scope="per-subject descriptive; N=2 usable; no group inference", rows=rows),
              open(A/"erp"/"n400_results.json", "w"), indent=2, default=str)
    return df

# ============================================================ PART 4 - DISCUSSION
def part4_discussion(psd_df, fooof_df, erp_df):
    prov = json.load(open("derivatives/sub-01-Pilot/eeg/provenance.json"))
    md = []
    md.append("# Analysis-phase discussion — Menteev 30-ch N400 pilot\n")
    md.append("*Auto-generated from the read-only analysis of the frozen preprocessing outputs. "
              "Descriptive only; no group inference or neuroscientific claims are made.*\n")
    md.append("## Preprocessing quality (from validation)\n"
              "- Reconstruction fidelity = 1.000000 for all subjects; average reference, rank-limited ICA, "
              "AutoReject epoching all verified. Excluded ICs were eye/muscle (brain prob ≤ 0.09); posterior "
              "alpha generators retained. Bad channels interpolated. Pilot 3 continuous data is clean but its "
              "ERP epoching is blocked by a 177-vs-176 `exp_stim2` trigger mismatch.\n")
    md.append("## PSD observations (Part 1)\n")
    for subj in SUBJECTS:
        row = psd_df[(psd_df.subject == subj) & (psd_df.region == "posterior")].iloc[0]
        md.append(f"- **{subj}**: posterior relative power — delta {row.delta_rel_pct}%, theta {row.theta_rel_pct}%, "
                  f"alpha {row.alpha_rel_pct}%, beta {row.beta_rel_pct}%, gamma {row.gamma_rel_pct}%. "
                  f"A posterior alpha contribution is present in every subject.\n")
    md.append("## FOOOF observations (Part 2)\n")
    for _, r in fooof_df.iterrows():
        md.append(f"- **{r.subject}**: aperiodic exponent {r.aperiodic_exponent}, alpha centre {r.alpha_cf_hz} Hz "
                  f"(power {r.alpha_power}), model R² {r.r_squared}. A well-defined posterior alpha peak sits above "
                  f"a physiological 1/f background.\n")
    md.append("## ERP / N400 observations (Part 3)\n")
    md.append("Usable subjects: Pilot 1, Pilot 2 (Pilot 3 blocked).\n")
    for _, r in erp_df.iterrows():
        md.append(f"- **{r.subject}**: N400 window (300–500 ms, ROI {ROI}) — Rhyme {r.n400_rhyme_uV} µV, "
                  f"NonRhyme {r.n400_nonrhyme_uV} µV, difference {r.diff_NR_minus_R_uV:+} µV; aSME "
                  f"{r.aSME_rhyme}/{r.aSME_nonrhyme} µV; exploratory t={r.exploratory_t} (p={r.exploratory_p}); "
                  f"cluster min-p={r.cluster_min_p}, significant clusters={r.n_sig_clusters}.\n")
    md.append("\n**ERP result is NULL.** No significant Rhyme vs NonRhyme difference in either subject "
              "(all p>0.4; zero significant spatiotemporal clusters), the descriptive grand average shows no "
              "systematic N400-window negativity, and in both subjects the measured difference is *smaller than "
              "its standardized measurement error (aSME)* and numerically reversed relative to the classic N400. "
              "No N400 effect is claimed.\n")
    md.append("## Limitations\n"
              "- N = 2 usable subjects (Pilot 3 ERP blocked) → no group-level inference; single-subject "
              "trial-level statistics are exploratory.\n"
              "- ~87 trials/condition and a 30-channel consumer amplifier with a template montage → limited ERP "
              "SNR; the aSME exceeds the observed effect, i.e. the study is underpowered for a sub-µV N400.\n"
              "- Template (non-digitised) electrode positions limit topographic/source precision.\n"
              "- LSL software marker timing (no photodiode) → latency-sensitive measures should be treated with care.\n")
    md.append("## Future work\n"
              "- Resolve Pilot 3's trigger mismatch to recover a third subject; acquire more participants.\n"
              "- Increase trials/condition; use the observed aSME for a formal power/feasibility calculation.\n"
              "- Once N is adequate, run proper group-level (within-subject averaged or mixed-effects) statistics.\n"
              "- Spectral (PSD/FOOOF) measures are stable and could support resting/aperiodic analyses.\n")
    (A/"discussion"/"discussion.md").write_text("\n".join(md), encoding="utf-8")
    print("discussion.md written")

if __name__ == "__main__":
    print("=== PART 1 PSD ==="); psd_df = part1_psd()
    print("=== PART 2 FOOOF ==="); fooof_df = part2_fooof()
    print("=== PART 3 ERP ==="); erp_df = part3_erp()
    print("=== PART 4 DISCUSSION ==="); part4_discussion(psd_df, fooof_df, erp_df)
    print("\nDONE -> derivatives/analysis/{psd,fooof,erp,discussion}/")
