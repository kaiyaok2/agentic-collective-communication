"""Curate confirmed 9-seed wins into CONFIRMED_WINS_ARTIFACTS/<group>/<problem>/:
  - all per-seed overlay + kiss code + json + logs (full provenance)
  - MEDIAN_SEED code for BOTH systems (the 5th-of-9 by sim time) -> E2E-ready
  - per-problem MANIFEST.json with the 9-seed stats and median-seed pointers
Also writes a group-level MANIFEST_9seed.json.
"""
import json, os, shutil, glob

FD = "/private/tmp/fair_diverge"
LEDGER = f"{FD}/campaign_ledger.json"
ART = f"{FD}/CONFIRMED_WINS_ARTIFACTS"

GROUPS = {
    "family1_9seed": ("r44fam1", [
        "r21_su8_countonly", "r23_deep8_count8", "r22_su8_count16", "r26_perm_count8",
        "r31_lin5_count8", "r20_su8_narr", "r16_deepdoc", "r1_fold_s4096",
        "r1_fold_s1024", "r2_deep4", "r1_fold_s256"]),
    "family2_9seed": ("r43fam2", [
        "r43_route_B11_d8_L3_count8", "r43_route_B10_strided_count8", "r43_route_B9_d8_L3_count8",
        "r40_route_d8_L3_count8", "r40_route_d6_L3_count8", "r43_route_B10_p384_count8",
        "r43_route_B12_d8_L3_res"]),
    "family3_9seed": ("r47fam3", [
        "r47_xc_r4", "r47_dd_relu_d8", "r47_dd_meanabs_d7", "r47_dd_meanabs_d8"]),
    "family3_9seed_r48": ("r48fam3", [
        "r48_dd_meansq_d8", "r48_dd_square_d8", "r48_dd_relu_d6"]),
}


def median_seed(seed_sims):
    """seed_sims: list of (seed, sim). Return the seed whose sim is the median
    (5th of 9 when sorted ascending)."""
    ok = [(s, v) for s, v in seed_sims if isinstance(v, (int, float))]
    ok.sort(key=lambda x: x[1])
    if not ok:
        return None, None
    mid = len(ok) // 2  # for odd n=9 -> index 4 = 5th smallest
    return ok[mid]


def kiss_sim(d):  # kiss_summary.json
    return d.get("best_sim_time_us")


def ov_sim(d):    # overlay.json
    return d.get("final_sim")


def curate():
    led = json.load(open(LEDGER))
    group_manifest = {}
    for group, (rnd, probs) in GROUPS.items():
        src_root = f"{FD}/results_{rnd}"
        gdir = f"{ART}/{group}"
        os.makedirs(gdir, exist_ok=True)
        gm = {}
        for p in probs:
            pdir = f"{gdir}/{p}"
            os.makedirs(pdir, exist_ok=True)
            # collect per-seed sims for median pick
            ov_seeds, ks_seeds = [], []
            for sd in sorted(glob.glob(f"{src_root}/{p}/overlay_s*")):
                if not os.path.isdir(sd):
                    continue
                seed = int(os.path.basename(sd).split("_s")[-1])
                oj = f"{sd}/overlay.json"
                if os.path.exists(oj):
                    ov_seeds.append((seed, ov_sim(json.load(open(oj)))))
                shutil.copytree(sd, f"{pdir}/{os.path.basename(sd)}", dirs_exist_ok=True)
            for sd in sorted(glob.glob(f"{src_root}/{p}/kiss_s*")):
                if not os.path.isdir(sd):
                    continue
                seed = int(os.path.basename(sd).split("_s")[-1])
                kj = f"{sd}/kiss_summary.json"
                if os.path.exists(kj):
                    ks_seeds.append((seed, kiss_sim(json.load(open(kj)))))
                shutil.copytree(sd, f"{pdir}/{os.path.basename(sd)}", dirs_exist_ok=True)
            # copy logs dir if present
            if os.path.isdir(f"{src_root}/{p}"):
                for lg in glob.glob(f"{src_root}/{p}/*.log"):
                    shutil.copy2(lg, pdir)
            # median-seed code for E2E
            ov_ms, ov_msv = median_seed(ov_seeds)
            ks_ms, ks_msv = median_seed(ks_seeds)
            e2e = f"{pdir}/MEDIAN_SEED_E2E"
            os.makedirs(e2e, exist_ok=True)
            if ov_ms is not None:
                oc = f"{src_root}/{p}/overlay_s{ov_ms}/best_code.py"
                if os.path.exists(oc):
                    shutil.copy2(oc, f"{e2e}/overlay_median_s{ov_ms}.py")
            if ks_ms is not None:
                kc = f"{src_root}/{p}/kiss_s{ks_ms}/best_code.py"
                if os.path.exists(kc):
                    shutil.copy2(kc, f"{e2e}/kiss_median_s{ks_ms}.py")
            conf = led.get("rounds", {}).get(rnd, {}).get("confirm", {}).get(p, {})
            gm[p] = {
                "confirmed": conf.get("CONFIRMED_DIVERGENCE"),
                "best_ratio": conf.get("best_ratio"),
                "median_ratio": conf.get("median_ratio"),
                "mannwhitney_p": conf.get("mannwhitney_p"),
                "ci95": conf.get("ci95"),
                "overlay_median_seed": ov_ms, "overlay_median_sim_us": ov_msv,
                "kiss_median_seed": ks_ms, "kiss_median_sim_us": ks_msv,
                "e2e_overlay_code": f"MEDIAN_SEED_E2E/overlay_median_s{ov_ms}.py",
                "e2e_kiss_code": f"MEDIAN_SEED_E2E/kiss_median_s{ks_ms}.py",
            }
            json.dump(gm[p], open(f"{pdir}/MANIFEST.json", "w"), indent=2)
            print(f"  [{group}] {p}: ov_median_s{ov_ms}={ov_msv} ks_median_s{ks_ms}={ks_msv} "
                  f"(median_ratio={conf.get('median_ratio')})")
        json.dump(gm, open(f"{gdir}/MANIFEST_9seed.json", "w"), indent=2)
        group_manifest[group] = {"round": rnd, "n_problems": len(probs), "problems": gm}
        print(f"[{group}] curated {len(probs)} problems -> {gdir}")
    json.dump(group_manifest, open(f"{ART}/MANIFEST_9seed_TOP.json", "w"), indent=2)
    print(f"\nTop manifest -> {ART}/MANIFEST_9seed_TOP.json")


if __name__ == "__main__":
    curate()
