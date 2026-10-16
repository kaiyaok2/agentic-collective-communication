#!/usr/bin/env python3
"""ncc_bench.py — reproducible measurement of neuronx-cc compilations.

For every (HLO module, flag set, compiler) triple it runs

    /usr/bin/time -v timeout <T> neuronx-cc compile --framework XLA --target trn1 <hlo> --output <neff> <flags>

in a fresh work directory and records: wall / user / sys seconds, peak RSS, exit status, PASS/FAIL/TIMEOUT/OOM
classification, compiler error code (NCC_*), NEFF size, HLO proto size, host load before the run, and the last lines of
the compiler log. Optionally executes the NEFF with `neuron-profile capture` and records the summary-json metrics
(device latency, MFU, DMA bytes, spills, ...).

Input manifest: JSONL rows with at least {"hlo_id":..., "hlo_pb": <path>} plus any metadata (copied to output rows).
Output: JSONL rows appended to --out (one row per compile), resumable via (hlo_id, flagset, cc_tag, repeat, cold).
"""
import argparse, json, os, re, shutil, subprocess, sys, time, hashlib, socket, platform

TIME = "/usr/bin/time"
FLAGSETS = {
    "default":      [],
    "O1":           ["--optlevel", "1"],
    "O3":           ["--optlevel", "3"],
    "transformer":  ["--model-type", "transformer"],
    "llm_training": ["--model-type", "transformer", "--distribution-strategy", "llm-training"],
    "autocast_bf16": ["--auto-cast", "all", "--auto-cast-type", "bf16"],
    "saturate":     ["--enable-saturate-infinity"],
    "mixed_acc":    ["--enable-mixed-precision-accumulation"],
    "fast_load":    ["--enable-fast-loading-neuron-binaries"],
}

def sh(cmd, **kw):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw)

def cc_version(cc_bin):
    r = sh([cc_bin, "--version"]); m = re.search(r"(\d+\.\d+\.\d+\.\d+\+[0-9a-f]+)", r.stdout)
    return m.group(1) if m else r.stdout.strip()[-60:]

def parse_time_v(txt):
    d = {}
    for line in txt.splitlines():
        if "Elapsed (wall clock)" in line:
            t = line.split(": ", 1)[1].strip(); parts = t.split(":")
            secs = 0.0
            for p in parts: secs = secs * 60 + float(p)
            d["wall_s"] = secs
        elif "User time (seconds)" in line: d["user_s"] = float(line.split(":")[-1])
        elif "System time (seconds)" in line: d["sys_s"] = float(line.split(":")[-1])
        elif "Maximum resident set size" in line: d["max_rss_kb"] = int(line.split(":")[-1])
        elif "Exit status" in line: d["exit_status"] = int(line.split(":")[-1])
        elif "Percent of CPU this job got" in line: d["cpu_percent"] = int(line.split(":")[-1].strip().rstrip("%") or 0)
        elif "Major (requiring I/O) page faults" in line: d["major_faults"] = int(line.split(":")[-1])
    return d

ERR_PATTERNS = [
    (r"(NCC_[A-Z0-9_]+)", "ncc_code"),
    (r"(ERROR .{0,160})", "error_line"),
    (r"(MemoryError|std::bad_alloc|Out of memory|Cannot allocate memory|Killed)", "oom_hint"),
    (r"(Compiler status (PASS|FAIL\w*))", "status_line"),
]

def classify(exit_status, timed_out, log_txt, neff_ok):
    info = {}
    for pat, key in ERR_PATTERNS:
        m = re.search(pat, log_txt)
        if m: info[key] = m.group(1)[:200]
    if timed_out: st = "TIMEOUT"
    elif neff_ok and exit_status == 0: st = "PASS"
    elif "oom_hint" in info or exit_status in (137, -9): st = "OOM"
    else: st = "FAIL"
    info["status"] = st
    return info

def drop_caches():
    subprocess.run("sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'", shell=True, check=False)

def run_one(cc_bin, hlo_pb, flags, workdir, timeout_s, target="trn1", verbose="warning"):
    os.makedirs(workdir, exist_ok=True)
    neff = os.path.join(workdir, "model.neff"); logf = os.path.join(workdir, "log-neuron-cc.txt"); tf = os.path.join(workdir, "time.txt")
    cmd = [TIME, "-v", "-o", tf, "timeout", "-k", "15", str(timeout_s), cc_bin, "compile", "--framework", "XLA", "--target", target,
           hlo_pb, "--output", neff, f"--verbose={verbose}", "--logfile", logf] + list(flags)
    load1 = os.getloadavg()[0]
    t0 = time.time(); r = sh(cmd, cwd=workdir); t1 = time.time()
    tv = parse_time_v(open(tf).read()) if os.path.exists(tf) else {}
    log_txt = r.stdout + ("\n" + open(logf, errors="ignore").read()[-6000:] if os.path.exists(logf) else "")
    exit_status = tv.get("exit_status", r.returncode)
    timed_out = exit_status in (124, 137) and (t1 - t0) >= timeout_s - 1
    neff_ok = os.path.exists(neff) and os.path.getsize(neff) > 0
    row = dict(tv); row.update(classify(exit_status, timed_out, log_txt, neff_ok))
    row.update(dict(outer_wall_s=round(t1 - t0, 3), neff_bytes=os.path.getsize(neff) if neff_ok else 0, load1_before=round(load1, 2),
                    log_tail=log_txt[-800:].replace("\n", " | ")))
    return row, neff if neff_ok else None

def profile_neff(neff, workdir, num_exec=10):
    """execute the NEFF standalone; return summary-json dict (first summary group) + capture wall time"""
    ntff = os.path.join(workdir, "profile.ntff"); out = os.path.join(workdir, "profile_summary.json")
    t0 = time.time()
    r = sh(["timeout", "-k", "10", "600", "neuron-profile", "capture", "-n", neff, "-s", ntff, "--num-exec", str(num_exec), "--ignore-exec-errors"], cwd=workdir)
    cap = time.time() - t0
    res = {"profile_capture_wall_s": round(cap, 2), "profile_capture_rc": r.returncode}
    if r.returncode != 0 or not os.path.exists(ntff):
        res["profile_err"] = r.stdout[-400:]; return res
    v = sh(["timeout", "-k", "10", "900", "neuron-profile", "view", "-n", neff, "-s", ntff, "--output-format", "summary-json", "--output-file", out], cwd=workdir)
    if v.returncode != 0 or not os.path.exists(out):
        res["profile_err"] = ("view: " + v.stdout[-400:]); return res
    try:
        d = json.load(open(out)); s = d.get("summary", d)
        if isinstance(s, list): s = s[0] if s else {}
        keep = {k: s[k] for k in s if isinstance(s[k], (int, float, str)) and len(str(s[k])) < 80}
        res["profile"] = keep
    except Exception as e:
        res["profile_err"] = f"parse: {e}"
    for f in (ntff, out):
        try: os.remove(f)
        except OSError: pass
    return res

def done_keys(out_path):
    keys = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                r = json.loads(line); keys.add((r["hlo_id"], r["flagset"], r["cc_tag"], r["repeat"], r["cold"]))
            except Exception: pass
    return keys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cc", default="neuronx-cc", help="path to neuronx-cc binary"); ap.add_argument("--cc-tag", default=None)
    ap.add_argument("--flagsets", default="default", help="comma list of FLAGSETS keys")
    ap.add_argument("--repeats", type=int, default=1); ap.add_argument("--cold", action="store_true", help="drop page caches before each compile")
    ap.add_argument("--timeout", type=int, default=1800); ap.add_argument("--profile", action="store_true", help="execute NEFF with neuron-profile")
    ap.add_argument("--num-exec", type=int, default=10); ap.add_argument("--workroot", default="/home/ubuntu/ncc_work")
    ap.add_argument("--keep-neff-dir", default=None, help="if set, copy PASS NEFFs here as <hlo_id>__<flagset>__<cc_tag>.neff")
    ap.add_argument("--filter", default=None, help="regex on hlo_id"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-hlo-bytes", type=int, default=0, help="skip HLO protos larger than this (0 = no limit)")
    a = ap.parse_args()
    cc_bin = shutil.which(a.cc) or a.cc; ver = cc_version(cc_bin); tag = a.cc_tag or ver.split("+")[0]
    rows = [json.loads(l) for l in open(a.manifest) if l.strip()]
    if a.filter: rows = [r for r in rows if re.search(a.filter, r["hlo_id"])]
    if a.max_hlo_bytes: rows = [r for r in rows if os.path.getsize(r["hlo_pb"]) <= a.max_hlo_bytes]
    if a.limit: rows = rows[:a.limit]
    done = done_keys(a.out); host = dict(hostname=socket.gethostname(), cpu=platform.processor() or platform.machine(), nproc=os.cpu_count())
    try: host["instance_type"] = sh(["bash", "-c", "TOKEN=$(curl -s -X PUT -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' http://169.254.169.254/latest/api/token); curl -s -H \"X-aws-ec2-metadata-token: $TOKEN\" http://169.254.169.254/latest/meta-data/instance-type"]).stdout.strip()[:40]
    except Exception: pass
    print(f"[bench] cc={cc_bin} ver={ver} tag={tag} rows={len(rows)} flagsets={a.flagsets} repeats={a.repeats} cold={a.cold} done={len(done)}", flush=True)
    n = 0
    with open(a.out, "a") as fo:
        for fs in a.flagsets.split(","):
            flags = FLAGSETS[fs]
            for rep in range(a.repeats):
                for r in rows:
                    key = (r["hlo_id"], fs, tag, rep, a.cold)
                    if key in done: continue
                    wd = os.path.join(a.workroot, f"{r['hlo_id']}__{fs}__{tag}__r{rep}{'c' if a.cold else ''}")
                    shutil.rmtree(wd, ignore_errors=True)
                    if a.cold: drop_caches()
                    row = dict(r); row.update(dict(flagset=fs, flags=flags, cc_tag=tag, cc_version=ver, repeat=rep, cold=a.cold, t_start=time.time(),
                                                   hlo_pb_bytes=os.path.getsize(r["hlo_pb"]), host=host))
                    res, neff = run_one(cc_bin, r["hlo_pb"], flags, wd, a.timeout)
                    row.update(res)
                    if neff and a.profile: row.update(profile_neff(neff, wd, a.num_exec))
                    if neff and a.keep_neff_dir:
                        os.makedirs(a.keep_neff_dir, exist_ok=True); shutil.copy(neff, os.path.join(a.keep_neff_dir, f"{r['hlo_id']}__{fs}__{tag}.neff"))
                    fo.write(json.dumps(row) + "\n"); fo.flush(); n += 1
                    print(f"[bench] {r['hlo_id'][:50]:50s} {fs:12s} {tag:8s} r{rep} {row['status']:7s} wall={row.get('wall_s', -1):8.2f}s rss={row.get('max_rss_kb', 0)/1e6:6.2f}GB neff={row['neff_bytes']/1e6:7.2f}MB {row.get('ncc_code', '')}", flush=True)
                    shutil.rmtree(wd, ignore_errors=True)
    print(f"[bench] wrote {n} rows to {a.out}", flush=True)

if __name__ == "__main__":
    main()
