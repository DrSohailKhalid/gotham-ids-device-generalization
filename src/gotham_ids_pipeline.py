# -*- coding: utf-8 -*-
"""Leakage-aware IoT intrusion detection evaluation on Gotham Dataset 2025.

Companion implementation for the manuscript:
"Beyond Random Splits: Generalization of IoT Intrusion Detection to Unseen Devices".

The pipeline acquires device-level CSVs, constructs causal features, runs three
validation protocols and exports fold-level measurements. Execution is designed
for Google Colab; configuration is provided immediately below.
"""

# ============================== CONFIG ===============================
SCALE = "main"            # "pilot" | "main" | "full"

ZENODO_RECORD = "14502760"          # GothamDataset2025
LABEL_GRANULARITY = "coarse6"       # "coarse6": the 6 categories the dataset authors
                                    #   themselves report (Fig. 4 / Table 4), so our
                                    #   numbers are directly comparable to theirs.
                                    # "as_is": all 10 raw label values.

USE_DRIVE = True                    # cache downloads + results to Google Drive
DRIVE_DIR = "/content/drive/MyDrive/ICIC2026_Gotham"
LOCAL_DIR = "/content/ICIC2026_Gotham"

SEED = 20260101

MODELS       = ["logreg", "rf", "histgb", "mlp"]
REGIME_NAMES = ["F-ALL", "F-NOID", "F-BEHAV"]
PROTOCOLS    = ["P1-random", "P2-device", "P3-segment"]

N_FOLDS_P1   = 5           # stratified random packet folds
P2_MAX_FOLDS = 8           # cap on leave-one-attacked-device-out folds

RUN_FEDERATED        = True    # Stage 7; set False to cut it
FL_ROUNDS            = 10      # matches the dataset authors' own setup
FL_CLIENTS_PER_ROUND = 13      # ditto
FL_EVAL_FOLDS        = 2       # how many P2 folds to run the FL arm on

# --- loader internals (leave alone unless a gate tells you otherwise) ---
CHUNK_ROWS      = 300_000
TAIL_BUFFER     = 2_000    # rows carried across chunks for rolling features
RARE_CHUNK_N    = 500      # labels rarer than this within a chunk are kept whole
MAX_FIT_MINUTES = 30       # print a SLOW warning past this
HGB_MAX_CARD    = 200      # max cardinality for a column to be declared categorical
# =====================================================================

_SCALES = {
    #        rows/device   devices (None = all 78)
    "pilot": (2_000,       26),     # smoke test: ~10-15 min after download
    "main":  (8_000,       None),   # reported benchmark configuration
    "full":  (40_000,      None),   # larger-scale confirmation
}
ROWS_PER_DEVICE, DEVICE_CAP = _SCALES[SCALE]

print(f"SCALE={SCALE}   ROWS_PER_DEVICE={ROWS_PER_DEVICE}   "
      f"DEVICE_CAP={DEVICE_CAP or 'all'}   SEED={SEED}")
print(f"labels={LABEL_GRANULARITY}")


# -------------------------------------------------------------------------
# Stage 0: Environment and determinism
# Record software versions, configuration identity, and seeded execution.
# -------------------------------------------------------------------------


import os, sys, json, time, hashlib, random, re, zipfile, shutil, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------- output tree
ROOT = Path(LOCAL_DIR)
if USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        ROOT = Path(DRIVE_DIR)
        print("Caching to Drive:", ROOT)
    except Exception as e:
        print("Drive unavailable, using local disk:", e)

CONFIG_HASH = hashlib.md5(json.dumps(
    {"scale": SCALE, "seed": SEED, "labels": LABEL_GRANULARITY,
     "models": sorted(MODELS), "regimes": REGIME_NAMES,
     "rows": ROWS_PER_DEVICE, "devcap": DEVICE_CAP},
    sort_keys=True).encode()).hexdigest()[:10]

RUN_TAG = f"{SCALE}_{LABEL_GRANULARITY}_{CONFIG_HASH}"
DATA    = ROOT / "data"                 # shared across runs: downloads are expensive
RAW     = DATA / "raw"                  # downloaded files
EXTRACT = DATA / "extracted"            # anything unpacked from an archive
RESULT  = ROOT / "results" / RUN_TAG    # per-config: checkpoints never mix
FIGS    = RESULT / "figures"
TABLES  = RESULT / "tables"
CONF    = RESULT / "confusion"
for d in (DATA, RAW, RESULT, FIGS, TABLES, CONF):
    d.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------- determinism
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

HAS_TORCH, DEVICE = False, "cpu"
try:
    import torch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    HAS_TORCH = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception as e:
    print("torch unavailable:", e)

import sklearn
env_lines = [
    f"python        {sys.version.split()[0]}",
    f"numpy         {np.__version__}",
    f"pandas        {pd.__version__}",
    f"scikit-learn  {sklearn.__version__}",
    f"torch         {(torch.__version__ if HAS_TORCH else 'absent')}",
    f"device        {DEVICE}",
    f"seed          {SEED}",
    f"scale         {SCALE} ({ROWS_PER_DEVICE} rows/device, devices={DEVICE_CAP or 'all'})",
    f"label_mode    {LABEL_GRANULARITY}",
    f"config_hash   {CONFIG_HASH}",
    f"run_tag       {RUN_TAG}",
]
(RESULT / "environment.txt").write_text("\n".join(env_lines))
print("\n".join(env_lines))


def gate(name, ok, detail=""):
    """Print a PASS/FAIL gate. Halt the run on FAIL."""
    print(f"GATE: {'PASS' if ok else 'FAIL'}  [{name}] {detail}")
    if not ok:
        raise SystemExit(f"Gate failed: {name}. Inspect the diagnostic before continuing.")
    return ok


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


gate("S0 environment", RESULT.exists(), f"outputs -> {RESULT}")


# -------------------------------------------------------------------------
# Stage 0b: Dataset constants
# Device families, network segments, column schema, and label normalization.
# -------------------------------------------------------------------------


# --- device -> family -> segment, from Fig. 4 of the data article ------
SEGMENT_MAP = {
    # segment 1 (9 devices)
    "air-quality": 1, "city-power": 1, "domotic-monitor": 1, "ip-camera-street": 1,
    # segment 2 (9 devices)
    "building-monitor": 2, "ip-camera-museum": 2, "stream-consumer": 2,
    # segment 3 (30 devices) -- benign only in this testbed
    "cooler-motor": 3, "predictive-maintenance": 3,
    # segment 4 (30 devices)
    "combined-cycle": 4, "combined-cycle-tls": 4, "hydraulic-system": 4,
}
EXPECTED_SEG_COUNTS = {1: 9, 2: 9, 3: 30, 4: 30}
BENIGN_ONLY_SEGMENT = 3
NORMAL = "Normal"

# Table 3 of the data article: instances per device type (78 in total).
EXPECTED_FAMILY_N = {
    "air-quality": 1, "building-monitor": 5, "city-power": 1, "combined-cycle": 10,
    "combined-cycle-tls": 5, "cooler-motor": 15, "domotic-monitor": 5,
    "hydraulic-system": 15, "ip-camera-street": 2, "ip-camera-museum": 2,
    "stream-consumer": 2, "predictive-maintenance": 15,
}

# Read off Fig. 4: the devices that appear to carry ANY attack traffic. This is a
# HINT used only to choose which files a pilot run downloads. The real inventory is
# measured from the data in Stage 2 and never depends on this list being right.
ATTACK_DEVICE_HINT = [
    "air-quality-1", "city-power-1", "domotic-monitor-1", "ip-camera-street-1",
    "building-monitor-5", "ip-camera-museum-1", "combined-cycle-1", "combined-cycle-2",
]

# Table 4 of the data article: the authors' own federated DNN, for comparison.
AUTHORS_DNN_F1 = {"Brute Force": 0.60, "C&C Communication": 0.97, "DoS": 0.94,
                  "Infection": 0.00, "Network Scanning": 0.56, "Normal": 0.69}
AUTHORS_DNN_ACC = 0.89


def device_name(name):
    """'iotsim-cooler-motor-3.csv' -> 'cooler-motor-3'"""
    s = str(name).split("/")[-1].lower()
    for suf in (".csv.gz", ".csv"):
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    return re.sub(r"^iotsim[-_]", "", s)


def device_family(dev):
    """Longest matching family prefix wins, so combined-cycle-tls-3 does not
    collapse into combined-cycle."""
    best = None
    for fam in SEGMENT_MAP:
        if dev == fam or dev.startswith(fam + "-") or dev.startswith(fam + "_"):
            if best is None or len(fam) > len(best):
                best = fam
    return best if best else re.sub(r"[-_]\d+$", "", dev)


def natkey(s):
    """Natural sort: cooler-motor-2 sorts before cooler-motor-10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


def select_devices(names, cap):
    """Pick <= cap device names, always keeping the hinted attack-bearing devices
    and at least one instance of every family, so all four segments stay present."""
    if not cap or cap >= len(names):
        return sorted(names, key=natkey)
    by_dev = {device_name(n): n for n in names}
    chosen = [d for d in ATTACK_DEVICE_HINT if d in by_dev]
    by_fam = defaultdict(list)
    for d in by_dev:
        by_fam[device_family(d)].append(d)
    for fam in sorted(by_fam):
        for d in sorted(by_fam[fam], key=natkey):
            if len(chosen) >= cap:
                break
            if d not in chosen:
                chosen.append(d)
                break
    pool = [d for d in sorted(by_dev, key=natkey) if d not in chosen]
    chosen += pool[: max(0, cap - len(chosen))]
    return sorted((by_dev[d] for d in chosen), key=natkey)


# --- column schema, Table 2 of the data article ------------------------
EXPECTED = ["frame.time", "frame.len", "frame.protocols", "eth.src", "eth.dst",
            "ip.dst", "ip.src", "ip.flags", "ip.ttl", "ip.proto", "ip.checksum",
            "ip.tos", "tcp.srcport", "tcp.dstport", "tcp.flags",
            "tcp.window_size_value", "tcp.window_size_scalefactor", "tcp.checksum",
            "tcp.options", "tcp.pdu.size", "udp.srcport", "udp.dstport", "label"]

ABS_TIME_COL = "_abs_time"   # parsed epoch seconds; NEVER a model feature


def canon(c):
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


CANON_TO_EXPECTED = {canon(c): c for c in EXPECTED}


def rename_columns(frame):
    ren = {c: CANON_TO_EXPECTED[canon(c)] for c in frame.columns
           if canon(c) in CANON_TO_EXPECTED}
    return frame.rename(columns=ren)


# --- label normalisation ----------------------------------------------
# The release's 10 raw values, per the article's Data Description section.
LABEL_ALIASES = {
    canon("Benign"): "Normal", canon("Normal"): "Normal",
    canon("DoS Attack"): "DoS", canon("DoS"): "DoS",
    canon("Network Scanning"): "Network Scanning",
    canon("Telnet Brute Forcing"): "Telnet Brute Force",
    canon("Telnet Brute Force"): "Telnet Brute Force",
    canon("Brute Force"): "Telnet Brute Force",
    canon("Reporting"): "Reporting",
    canon("Ingress Tool Transfer"): "Ingress Tool Transfer",
    canon("File Download"): "Ingress Tool Transfer",
    canon("Remote Command Execution"): "Remote Command Execution",
    canon("Remote Code Execution"): "Remote Code Execution",
    canon("Periodic C&C Communication"): "Periodic C&C Communication",
    canon("C&C Communication"): "Periodic C&C Communication",
    canon("CoAP Amplification Attack"): "CoAP Amplification",
    canon("CoAP Amplification"): "CoAP Amplification",
    canon("Infection"): "Infection",
    # --- added: raw values present in the CURRENT Zenodo release that were
    # not yet in the original Data in Brief article's label list. The record
    # has clearly been revised/expanded since publication (finer flooding
    # and scan subtypes, separate Merlin vs Mirai C&C). Verified against an
    # actual pull of the live data (18 raw label strings observed).
    canon("TCP Scan"): "TCP Scan",
    canon("UDP Scan"): "UDP Scan",
    canon("Mirai UDP Flooding"): "Mirai UDP Flooding",
    canon("Mirai TCP Flooding"): "Mirai TCP Flooding",
    canon("Mirai GRE Flooding"): "Mirai GRE Flooding",
    canon("Merlin TCP Flooding"): "Merlin TCP Flooding",
    canon("Merlin UDP Flooding"): "Merlin UDP Flooding",
    canon("Merlin ICMP Flooding"): "Merlin ICMP Flooding",
    canon("Merlin C&C Communication"): "Merlin C&C Communication",
    canon("Mirai C&C Communication"): "Mirai C&C Communication",
    canon("Unknown"): "Unknown",
}
# The authors' own 6 categories (Fig. 4, Table 4), extended with the
# newer raw labels above, grouped by attack semantics (flooding -> DoS,
# scan subtypes -> Network Scanning, C&C variants -> C&C Communication).
# "Unknown" is kept as its own 7th bucket rather than folded into an
# existing class, since silently merging unlabelled packets into a real
# attack/benign category would bias that class's reported performance.
COARSE6 = {
    "Normal": "Normal",
    "DoS": "DoS",
    "CoAP Amplification": "DoS",
    "Mirai UDP Flooding": "DoS",
    "Mirai TCP Flooding": "DoS",
    "Mirai GRE Flooding": "DoS",
    "Merlin TCP Flooding": "DoS",
    "Merlin UDP Flooding": "DoS",
    "Merlin ICMP Flooding": "DoS",
    "Network Scanning": "Network Scanning",
    "TCP Scan": "Network Scanning",
    "UDP Scan": "Network Scanning",
    "Telnet Brute Force": "Brute Force",
    "Reporting": "Infection",
    "Ingress Tool Transfer": "Infection",
    "Infection": "Infection",
    "Remote Command Execution": "C&C Communication",
    "Remote Code Execution": "C&C Communication",
    "Periodic C&C Communication": "C&C Communication",
    "Merlin C&C Communication": "C&C Communication",
    "Mirai C&C Communication": "C&C Communication",
    "Unknown": "Unknown",
}
EXPECTED_LABELS = (sorted(set(COARSE6.values())) if LABEL_GRANULARITY == "coarse6"
                   else sorted(set(LABEL_ALIASES.values())))
UNKNOWN_LABELS = set()
_LBL_CACHE = {}


def normalise_label(v):
    if v in _LBL_CACHE:
        return _LBL_CACHE[v]
    out = LABEL_ALIASES.get(canon(v))
    if out is None:
        UNKNOWN_LABELS.add(str(v))
        out = str(v).strip().title()
    if LABEL_GRANULARITY == "coarse6":
        out = COARSE6.get(out, out)
    _LBL_CACHE[v] = out
    return out


print("families mapped :", len(SEGMENT_MAP))
print("expected devices:", sum(EXPECTED_SEG_COUNTS.values()))
print("expected labels :", EXPECTED_LABELS)
gate("S0b constants", sum(EXPECTED_FAMILY_N.values()) == 78
     and sum(EXPECTED_SEG_COUNTS.values()) == 78,
     "device counts from Tables 3 and Fig. 4 reconcile to 78")

# Drive mounting is handled once during environment initialization.


# -------------------------------------------------------------------------
# Stage 1: Dataset acquisition
# Download the device-level CSVs from the Zenodo release and verify checksums.
# -------------------------------------------------------------------------


import requests

API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
resp = requests.get(API, timeout=180)
resp.raise_for_status()
rec = resp.json()


def _entries(record):
    """Zenodo legacy and InvenioRDM response shapes both occur in the wild."""
    f = record.get("files", [])
    out = []
    if isinstance(f, dict) and "entries" in f:
        for k, v in f["entries"].items():
            out.append({"key": k, "size": v.get("size"),
                        "checksum": v.get("checksum") or "",
                        "link": (v.get("links") or {}).get("content")})
    else:
        for v in f:
            out.append({"key": v.get("key"), "size": v.get("size"),
                        "checksum": v.get("checksum") or "",
                        "link": (v.get("links") or {}).get("self")})
    return [e for e in out if e["key"] and e["link"]]


entries = _entries(rec)
print("record title :", (rec.get("metadata") or {}).get("title", "?"))
print("published    :", (rec.get("metadata") or {}).get("publication_date", "?"))
print(f"files in record: {len(entries)}\n")

by_size = sorted(entries, key=lambda x: -(x["size"] or 0))
for e in by_size[:25]:
    print(f"  {e['key'][:66]:<66} {(e['size'] or 0)/1e6:10.1f} MB")
if len(by_size) > 25:
    print(f"  ... and {len(by_size)-25} more")


def wanted(key):
    k = key.lower()
    if "pcap" in k or k.endswith((".pcap", ".pcapng")):
        return False
    return k.endswith((".csv", ".csv.gz", ".zip", ".tar.gz", ".tgz", ".7z"))


cands = [e for e in entries if wanted(e["key"])]
csv_entries = [e for e in cands if e["key"].lower().endswith((".csv", ".csv.gz"))]
arch_entries = [e for e in cands if e not in csv_entries]

if csv_entries and DEVICE_CAP:
    keep = set(select_devices([e["key"] for e in csv_entries], DEVICE_CAP))
    SELECTED = [e for e in csv_entries if e["key"] in keep]
    MODE = f"per-device CSVs, pilot subset of {len(SELECTED)}/{len(csv_entries)}"
elif csv_entries:
    SELECTED = csv_entries
    MODE = f"per-device CSVs, all {len(csv_entries)}"
else:
    SELECTED = arch_entries          # the release ships archives: take them all
    MODE = f"{len(arch_entries)} archive(s); device subset applied after extraction"

tot_gb = sum(e["size"] or 0 for e in SELECTED) / 1e9
print(f"\n{MODE}")
print(f"{len(SELECTED)} files to fetch, {tot_gb:.2f} GB")
gate("S1 candidate files", len(SELECTED) > 0, f"{len(SELECTED)} files, {tot_gb:.2f} GB")

def md5_of(path, buf=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(buf), b""):
            h.update(blk)
    return h.hexdigest()


def fetch(entry):
    """Download with resume-by-size; a completed file is never re-fetched."""
    dest = RAW / entry["key"].replace("/", "_")
    if dest.exists() and entry["size"] and dest.stat().st_size == entry["size"]:
        return dest, "cached"
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(entry["link"], stream=True, timeout=900) as r:
        r.raise_for_status()
        done, nxt = 0, 1 << 28
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if done >= nxt:
                    print(f"    {entry['key'][:48]}: {done/1e6:,.0f} MB", flush=True)
                    nxt += 1 << 28
    tmp.rename(dest)
    return dest, "downloaded"


# Reuse a previous manifest so a cached, already-verified file is never
# re-hashed on restart -- md5'ing several GB on every run was the actual
# slow part here, not the (cache-skipped) download itself.
_prev_manifest_path = RESULT / "data_manifest.csv"
_prev = {}
if _prev_manifest_path.exists():
    for r in pd.read_csv(_prev_manifest_path).to_dict("records"):
        _prev[r["file"]] = r

manifest, t0 = [], time.time()
for i, e in enumerate(SELECTED, 1):
    print(f"[{i}/{len(SELECTED)}] {e['key'][:60]}", flush=True)
    path, how = fetch(e)
    expect = (e["checksum"] or "").split(":")[-1]
    size_now = path.stat().st_size
    prior = _prev.get(e["key"])
    reuse = (how == "cached" and prior is not None
             and int(prior.get("bytes", -1)) == size_now
             and bool(prior.get("verified", False)))
    if reuse:
        got, ok = True, ""
        how = "cached-verified (skipped re-hash)"
    else:
        got = md5_of(path) if expect else ""
        ok = (not expect) or (got == expect)
    manifest.append({"file": e["key"], "local": str(path), "bytes": size_now,
                     "md5_expected": expect, "md5_actual": got,
                     "verified": bool(ok), "how": how})
    print(f"    {how}, md5 {'OK' if ok else '*** MISMATCH ***'}", flush=True)

man = pd.DataFrame(manifest)
man.to_csv(RESULT / "data_manifest.csv", index=False)
print(f"\nacquisition finished in {(time.time()-t0)/60:.1f} min")
gate("S1 checksums", bool(man["verified"].all()),
     f"{len(man)} files, {int(man['verified'].sum())} verified")

# Expand any archive, then collect the per-device CSV paths.
for p in sorted(list(RAW.glob("*.zip")) + list(RAW.glob("*.tar.gz")) + list(RAW.glob("*.tgz"))):
    marker = EXTRACT / (p.name + ".done")
    if marker.exists():
        continue
    EXTRACT.mkdir(exist_ok=True)
    print("extracting", p.name, flush=True)
    if p.suffix == ".zip":
        with zipfile.ZipFile(p) as z:
            z.extractall(EXTRACT)
    else:
        import tarfile
        with tarfile.open(p) as t:
            t.extractall(EXTRACT)
    marker.write_text("ok")

found = list(RAW.glob("*.csv")) + list(RAW.glob("*.csv.gz"))
if EXTRACT.exists():
    found += list(EXTRACT.rglob("*.csv")) + list(EXTRACT.rglob("*.csv.gz"))
ALL_CSV = sorted({p for p in found if "pcap" not in p.name.lower()}, key=natkey)

# The archive path downloads everything, so the pilot subset is applied here instead.
keep = set(select_devices([str(p) for p in ALL_CSV], DEVICE_CAP))
CSV_PATHS = [p for p in ALL_CSV if str(p) in keep]
SUBSET = len(CSV_PATHS) < len(ALL_CSV)

print(f"{len(ALL_CSV)} device CSVs on disk, {len(CSV_PATHS)} selected "
      f"(subset={SUBSET})")
fam_counts = pd.Series([device_family(device_name(p.name)) for p in CSV_PATHS])
print(fam_counts.value_counts().sort_index().to_string())
missing_hint = [d for d in ATTACK_DEVICE_HINT
                if d not in {device_name(p.name) for p in CSV_PATHS}]
if missing_hint:
    print("\nnote: hinted attack devices not present:", missing_hint)

gate("S1 device count", len(CSV_PATHS) >= 8 and (len(ALL_CSV) >= 70 or SUBSET),
     f"{len(ALL_CSV)} CSVs on disk (the release documents 78 devices), "
     f"{len(CSV_PATHS)} in use")


# -------------------------------------------------------------------------
# Stage 2: Working benchmark
# Stream and subsample per-device traffic while preserving causal feature order.
# -------------------------------------------------------------------------


# ---- behavioural feature derivation (causal, per device, order-preserving)
SERVICE_PORTS = {"is_mqtt": (1883, 8883), "is_coap": (5683, 5684), "is_rtsp": (554,),
                 "is_telnet": (23,), "is_dns": (53,), "is_http": (80, 443)}
STACK_TOKENS = {"has_tcp": "tcp", "has_udp": "udp", "has_icmp": "icmp",
                "has_mqtt": "mqtt", "has_coap": "coap", "has_rtsp": "rtsp",
                "has_tls": "tls", "has_dns": "dns"}
DERIVED = (["iat", "log_iat", "rate_1s", "rate_5s", "len_mean_20", "len_std_20",
            "srcport_bucket", "dstport_bucket"]
           + list(STACK_TOKENS) + list(SERVICE_PORTS))


def parse_time(s, n):
    """Return epoch seconds as float64. Falls back to packet index."""
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.9 and float(num.max() or 0) > 1e8:
        return num.astype("float64").to_numpy()
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    if dt.notna().mean() > 0.5:
        vals = dt.astype("int64").astype("float64") / 1e9
        vals[dt.isna().to_numpy()] = np.nan
        vals = pd.Series(vals).ffill().bfill().fillna(0.0).to_numpy()
        return vals
    return np.arange(n, dtype="float64")


def _num(frame, col):
    """Numeric view of a column that may be absent."""
    if col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def port_bucket(p):
    v = p.fillna(-1).to_numpy()
    return np.select([v < 0, v < 1024, v < 49152], [0, 1, 2], default=3).astype("int8")


def add_derived(frame, tail):
    """frame: one chunk in packet order. tail: previous chunk's last rows."""
    work = (pd.concat([tail, frame], ignore_index=True)
            if tail is not None and len(tail) else frame.reset_index(drop=True))
    n_tail = 0 if tail is None else len(tail)

    t = work[ABS_TIME_COL].to_numpy(dtype="float64")
    iat = np.diff(t, prepend=(t[0] if len(t) else 0.0))
    iat = np.clip(np.nan_to_num(iat, nan=0.0, posinf=0.0, neginf=0.0), 0, None)
    work["iat"] = iat.astype("float32")
    work["log_iat"] = np.log1p(iat).astype("float32")

    # trailing-window packet rate: causal, uses only packets at or before t
    order = np.argsort(t, kind="stable")
    ts = t[order]
    for win, col in ((1.0, "rate_1s"), (5.0, "rate_5s")):
        left = np.searchsorted(ts, ts - win, side="left")
        cnt = (np.arange(len(ts)) - left + 1).astype("float32")
        out = np.empty(len(ts), dtype="float32")
        out[order] = cnt
        work[col] = out

    ln = _num(work, "frame.len").astype("float32")
    work["len_mean_20"] = ln.rolling(20, min_periods=1).mean().astype("float32")
    work["len_std_20"] = ln.rolling(20, min_periods=1).std().fillna(0).astype("float32")

    sp = _num(work, "tcp.srcport").fillna(_num(work, "udp.srcport"))
    dp = _num(work, "tcp.dstport").fillna(_num(work, "udp.dstport"))
    work["srcport_bucket"] = port_bucket(sp)
    work["dstport_bucket"] = port_bucket(dp)
    for name, ports in SERVICE_PORTS.items():
        work[name] = (sp.isin(ports) | dp.isin(ports)).astype("int8")

    if "frame.protocols" in work.columns:
        proto = work["frame.protocols"].astype(str).str.lower()
    else:
        proto = pd.Series([""] * len(work), index=work.index)
    for name, tok in STACK_TOKENS.items():
        work[name] = proto.str.contains(tok, regex=False).astype("int8")

    return work.iloc[n_tail:].reset_index(drop=True)

def bytes_per_row(path, probe=4 << 20):
    """Estimate mean bytes per CSV line from the file head. Cheap and accurate enough."""
    opener = __import__("gzip").open if str(path).endswith(".gz") else open
    with opener(path, "rb") as fh:
        head = fh.read(probe)
    nl = head.count(b"\n")
    return (len(head) / nl) if nl > 5 else 200.0


def read_header(path, nrows=5):
    """Detect whether the CSV carries a header row.

    Returns the value to pass to read_csv's `header`: 0 when row 0 holds the
    column names, None when the file is headerless (then `names` is supplied).
    """
    probe = pd.read_csv(path, nrows=nrows, low_memory=False)
    hits = sum(1 for c in probe.columns if canon(c) in CANON_TO_EXPECTED)
    return (0 if hits >= 5 else None), hits


def sample_chunk(frame, frac, rng):
    """Keep every row of a label that is rare inside this chunk; subsample the rest."""
    if frac >= 1.0:
        return frame
    counts = frame["label"].value_counts()
    rare = set(counts[counts < RARE_CHUNK_N].index)
    keep = frame[frame["label"].isin(rare)]
    rest = frame[~frame["label"].isin(rare)]
    if len(rest):
        rest = rest.sample(frac=frac, random_state=rng)
    return pd.concat([keep, rest], ignore_index=True)


def cap_device(frame, n, rng):
    """Final per-device cap. Rare classes are protected from the cap."""
    if len(frame) <= n:
        return frame
    counts = frame["label"].value_counts()
    floor = max(50, n // 50)
    rare = set(counts[counts < floor].index)
    keep = frame[frame["label"].isin(rare)]
    rest = frame[~frame["label"].isin(rare)]
    room = max(0, n - len(keep))
    if room <= 0 or not len(rest):
        return keep.reset_index(drop=True)
    frac = min(1.0, room / len(rest))
    picks = [g.sample(max(1, int(round(len(g) * frac))), random_state=rng)
             for _, g in rest.groupby("label", sort=False)]
    return pd.concat([keep] + picks, ignore_index=True)


def load_device(path, rng):
    dev = device_name(path)
    fam = device_family(dev)
    seg = SEGMENT_MAP.get(fam, -1)
    size = path.stat().st_size
    est_rows = max(1.0, size / bytes_per_row(path))
    frac = min(1.0, ROWS_PER_DEVICE / est_rows)
    hdr, hits = read_header(path)
    names = EXPECTED if hdr is None else None

    kept, tail, rows_seen = [], None, 0
    label_seen = defaultdict(int)
    reader = pd.read_csv(path, chunksize=CHUNK_ROWS, low_memory=False,
                         header=hdr, names=names)
    for chunk in reader:
        chunk = rename_columns(chunk)
        if "label" not in chunk.columns:
            raise ValueError(f"{path.name}: no label column. "
                             f"Columns seen: {list(chunk.columns)[:25]}")
        chunk["label"] = chunk["label"].map(normalise_label)
        src = chunk["frame.time"] if "frame.time" in chunk.columns else pd.Series(dtype=object)
        chunk[ABS_TIME_COL] = parse_time(src, len(chunk))
        chunk = chunk.sort_values(ABS_TIME_COL, kind="stable").reset_index(drop=True)
        chunk = add_derived(chunk, tail)
        tail = chunk.tail(TAIL_BUFFER).copy()
        rows_seen += len(chunk)
        for k, v in chunk["label"].value_counts().items():
            label_seen[k] += int(v)
        kept.append(sample_chunk(chunk, frac, rng))

    out = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()
    out = cap_device(out, ROWS_PER_DEVICE, rng)
    out["device"] = dev
    out["device_family"] = fam
    out["segment"] = seg
    n_attack = int(sum(v for k, v in label_seen.items() if k != NORMAL))
    return out, {"device": dev, "family": fam, "segment": seg,
                 "rows_total": rows_seen, "rows_kept": len(out),
                 "attack_rows_total": n_attack,
                 "attack_share": round(n_attack / max(1, rows_seen), 5),
                 "header_hits": hits, "file_mb": round(size / 1e6, 1)}


print("loader ready")

TABLE_PATH = DATA / f"working_table_{RUN_TAG}.parquet"
STATS_PATH = RESULT / "table_stats.csv"

if TABLE_PATH.exists() and STATS_PATH.exists():
    df = pd.read_parquet(TABLE_PATH)
    stats = pd.read_csv(STATS_PATH)
    print("loaded cached working table:", df.shape)
else:
    rng = np.random.RandomState(SEED)
    frames, stat_rows, t0 = [], [], time.time()
    for i, p in enumerate(CSV_PATHS, 1):
        sub, st = load_device(p, rng)
        frames.append(sub)
        stat_rows.append(st)
        print(f"[{i:>3}/{len(CSV_PATHS)}] {st['device']:<32} seg={st['segment']} "
              f"{st['file_mb']:>8.1f}MB  {st['rows_total']:>10,} -> {st['rows_kept']:>7,}  "
              f"atk={st['attack_share']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    df = pd.concat(frames, ignore_index=True)
    stats = pd.DataFrame(stat_rows)
    stats.to_csv(STATS_PATH, index=False)
    df.to_parquet(TABLE_PATH, index=False)
    print(f"\nworking table: {df.shape}  built in {(time.time()-t0)/60:.1f} min")

if UNKNOWN_LABELS:
    print("\n*** UNRECOGNISED LABEL STRINGS -- report these: ***",
          sorted(UNKNOWN_LABELS))
print()
print(df["label"].value_counts().to_string())

#---- segment / family reconciliation ---------------------------------
seg_counts = {int(k): int(v) for k, v
              in stats.groupby("segment")["device"].nunique().items()}
unmapped = sorted(stats.loc[stats["segment"] < 0, "device"].tolist())
print("segment device counts:", seg_counts)
print("expected (full release):", EXPECTED_SEG_COUNTS)
if unmapped:
    print("*** UNMAPPED DEVICES:", unmapped)

if SUBSET:
    ok_seg = (not unmapped) and set(seg_counts) == set(EXPECTED_SEG_COUNTS)
    detail = f"subset run: {seg_counts}, all 4 segments present"
else:
    ok_seg = (not unmapped) and all(
        seg_counts.get(k) == v for k, v in EXPECTED_SEG_COUNTS.items())
    detail = f"{seg_counts} vs expected {EXPECTED_SEG_COUNTS}"
gate("S2 segments", ok_seg, detail)

present = set(df["label"].unique())
missing = [c for c in EXPECTED_LABELS if c not in present]
extra = sorted(present - set(EXPECTED_LABELS))
print("labels present:", sorted(present))
if extra:
    print("*** unexpected labels:", extra)
gate("S2 labels", len(present) >= 2 and not extra,
     f"{len(present)}/{len(EXPECTED_LABELS)} expected classes; missing={missing}")

# ---- attack-bearing device inventory
dev_lab = pd.crosstab(df["device"], df["label"])
for c in EXPECTED_LABELS:
    if c not in dev_lab.columns:
        dev_lab[c] = 0
dev_lab = dev_lab[EXPECTED_LABELS]
dev_share = (dev_lab.div(dev_lab.sum(1).clip(lower=1), axis=0) * 100).round(2)
dev_share["segment"] = stats.set_index("device")["segment"].reindex(dev_share.index)
dev_share["attack_pct"] = (100 - dev_share.get(NORMAL, 0)).round(2)
dev_share = dev_share.sort_values(["segment", "attack_pct"], ascending=[True, False])
dev_share.to_csv(TABLES / "table1b_device_label_share.csv")

ATTACK_DEVICES = sorted(dev_share.index[dev_share["attack_pct"] > 0].tolist())
BENIGN_DEVICES = sorted(d for d in dev_share.index if d not in ATTACK_DEVICES)
save_json({"attack_devices": ATTACK_DEVICES, "benign_devices": BENIGN_DEVICES},
          RESULT / "device_roles.json")

print(f"\n>>> {len(ATTACK_DEVICES)} of {dev_share.shape[0]} devices carry ANY attack traffic")
print(dev_share.loc[ATTACK_DEVICES, ["segment", "attack_pct"]].to_string())
gate("S2 attack devices", len(ATTACK_DEVICES) >= 2,
     f"{len(ATTACK_DEVICES)} attack-bearing devices -> P2 folds")

# ---- Table I + Fig. 0 -------------------------------------------------
comp = (pd.crosstab(df["segment"], df["label"])
        .reindex(columns=EXPECTED_LABELS, fill_value=0))
comp["devices"] = stats.groupby("segment")["device"].nunique()
comp["rows"] = df.groupby("segment").size()
comp.to_csv(TABLES / "table1_composition.csv")
print(comp.to_string())

piv = (pd.crosstab(df["device_family"], df["label"], normalize="index") * 100)
piv = piv.reindex(columns=EXPECTED_LABELS, fill_value=0).round(1)
fig, ax = plt.subplots(figsize=(9, 4.5))
im = ax.imshow(piv.values, aspect="auto", vmin=0, vmax=100)
ax.set_xticks(range(piv.shape[1]))
ax.set_xticklabels(piv.columns, rotation=40, ha="right", fontsize=8)
ax.set_yticks(range(piv.shape[0]))
ax.set_yticklabels(piv.index, fontsize=8)
ax.set_title("Label share (%) by device family, after per-device sampling")
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                fontsize=6, color="w" if v < 60 else "k")
fig.colorbar(im, ax=ax, shrink=0.85)
fig.tight_layout()
fig.savefig(FIGS / "fig0_label_by_device.png", dpi=600, bbox_inches="tight")
fig.savefig(FIGS / "fig0_label_by_device.pdf", dpi=600, bbox_inches="tight")
plt.close(fig)
print("\nFig. 0 written")


# -------------------------------------------------------------------------
# Stage 3: Hygiene and leakage audit
# Inspect duplicate records and the predictive power of shortcut fields.
# -------------------------------------------------------------------------


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, balanced_accuracy_score, matthews_corrcoef,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight

FEATURES_RAW = [c for c in EXPECTED if c != "label" and c in df.columns]
CATEGORICAL = [c for c in ["frame.protocols", "eth.src", "eth.dst", "ip.src", "ip.dst",
                           "ip.flags", "ip.proto", "tcp.flags", "tcp.options"]
               if c in df.columns]
print(f"{len(FEATURES_RAW)} raw features, {len(CATEGORICAL)} categorical")

# ---- duplicates -------------------------------------------------------
hash_parts = []
for c in FEATURES_RAW:
    hash_parts.append(pd.util.hash_pandas_object(df[c].astype(str), index=False)
                      .to_numpy(dtype="uint64"))
key = np.zeros(len(df), dtype="uint64")
for h in hash_parts:
    key = key * np.uint64(1000003) + h
del hash_parts

dup_rate = float(pd.Series(key).duplicated(keep="first").mean())
grp = pd.DataFrame({"k": key, "y": df["label"].to_numpy()})
ncls_per_key = grp.groupby("k")["y"].nunique()
cross_rate = float((ncls_per_key > 1).sum() / max(1, len(ncls_per_key)))
const_cols = [c for c in FEATURES_RAW if df[c].nunique(dropna=False) <= 1]

hyg = pd.DataFrame([{
    "rows": len(df), "features": len(FEATURES_RAW),
    "exact_duplicate_rate": round(dup_rate, 4),
    "cross_label_duplicate_group_rate": round(cross_rate, 4),
    "constant_columns": ";".join(const_cols) or "none",
}])
hyg.to_csv(RESULT / "hygiene.csv", index=False)
print(hyg.T.to_string(header=False))

# ---- shared evaluation subsample --------------------------------------
AUDIT_N = min(200_000, len(df))
sub_idx = np.random.RandomState(SEED).choice(len(df), size=AUDIT_N, replace=False)
ysub = df["label"].to_numpy()[sub_idx]
print(f"audit subsample: {AUDIT_N:,} rows, {len(set(ysub))} classes")


def encode_1d(col):
    s = df[col].iloc[sub_idx]
    if col in CATEGORICAL or s.dtype == object:
        return pd.factorize(s.astype(str))[0].reshape(-1, 1).astype("float64")
    return pd.to_numeric(s, errors="coerce").fillna(-1).to_numpy().reshape(-1, 1)


rows = []
for c in FEATURES_RAW:
    X1 = encode_1d(c)
    Xtr, Xte, ytr, yte = train_test_split(X1, ysub, test_size=0.3,
                                          random_state=SEED, stratify=ysub)
    r = {"feature": c, "n_unique": int(df[c].nunique(dropna=False))}
    for depth, name in ((3, "macroF1_depth3"), (12, "macroF1_depth12")):
        t = DecisionTreeClassifier(max_depth=depth, random_state=SEED,
                                   class_weight="balanced").fit(Xtr, ytr)
        r[name] = round(f1_score(yte, t.predict(Xte), average="macro",
                                 zero_division=0), 4)
    rows.append(r)

audit = pd.DataFrame(rows).sort_values("macroF1_depth12", ascending=False)
audit.to_csv(RESULT / "leakage_audit.csv", index=False)
print(audit.to_string(index=False))

# ---- identifier-only / time-only probes -------------------------------
def probe(cols, name, use_abs_time=False):
    cols = [c for c in cols if c in df.columns]
    if use_abs_time:
        X = df[ABS_TIME_COL].to_numpy()[sub_idx].reshape(-1, 1)
        shown = ABS_TIME_COL
    elif cols:
        X = np.hstack([encode_1d(c) for c in cols])
        shown = ",".join(cols)
    else:
        return {"probe": name, "features": "", "macroF1": np.nan,
                "balanced_acc": np.nan, "accuracy": np.nan}
    Xtr, Xte, ytr, yte = train_test_split(X, ysub, test_size=0.3,
                                          random_state=SEED, stratify=ysub)
    clf = RandomForestClassifier(n_estimators=150, random_state=SEED, n_jobs=-1,
                                 class_weight="balanced_subsample").fit(Xtr, ytr)
    pred = clf.predict(Xte)
    return {"probe": name, "features": shown,
            "macroF1": round(f1_score(yte, pred, average="macro", zero_division=0), 4),
            "balanced_acc": round(balanced_accuracy_score(yte, pred), 4),
            "accuracy": round(float((pred == yte).mean()), 4)}


probes = pd.DataFrame([
    probe(["ip.src", "ip.dst", "eth.src", "eth.dst"], "identifiers only"),
    probe([], "absolute time only", use_abs_time=True),
    probe(["ip.checksum", "tcp.checksum"], "checksums only"),
    probe(["frame.len", "tcp.flags", "ip.ttl", "ip.proto"], "4 behavioural (reference)"),
])
probes.to_csv(TABLES / "table2_probes.csv", index=False)
print(probes.to_string(index=False))

gate("S3 probes", bool(probes["macroF1"].notna().any()), "probes completed")

id_f1 = float(probes.loc[0, "macroF1"])
if id_f1 < 0.5:
    print("\n*** NOTE: the identifier probe is WEAK on this release (macro-F1 "
          f"{id_f1:.3f}). Review the shortcut-learning interpretation "
          "before reporting results. ***")
else:
    print(f"\nIdentifiers alone reach macro-F1 {id_f1:.3f} with no behavioural "
          "features at all.")

# ---- Fig. 1 -----------------------------------------------------------
top = audit.head(14).iloc[::-1]
chance = 1.0 / max(2, df["label"].nunique())
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.barh(top["feature"], top["macroF1_depth12"])
for bar, val in zip(bars, top["macroF1_depth12"]):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", ha="left", va="center", fontsize=7)
ax.axvline(chance, linestyle="--", linewidth=1, color="0.4")
ax.text(chance, -0.7, " chance", fontsize=7, color="0.4", va="top")
ax.set_xlabel("macro-F1 from this feature alone (depth-12 tree)")
ax.set_xlim(0, 1.08)
ax.set_title("Single-feature discriminability, Gotham 2025")
fig.tight_layout()
fig.savefig(FIGS / "fig1_leakage.png", dpi=600, bbox_inches="tight")
fig.savefig(FIGS / "fig1_leakage.pdf", dpi=600, bbox_inches="tight")
plt.close(fig)
print("Fig. 1 written")


# -------------------------------------------------------------------------
# Stage 4: Feature regimes
# Specify raw, identifier-restricted, and behavioural representations.
# -------------------------------------------------------------------------


IDENTIFIERS   = ["eth.src", "eth.dst", "ip.src", "ip.dst"]
FINGERPRINTS  = ["ip.checksum", "tcp.checksum"]
ABSOLUTE_TIME = ["frame.time", ABS_TIME_COL]
RAW_PORTS     = ["tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport"]
BANNED_BEHAV  = set(IDENTIFIERS + FINGERPRINTS + ABSOLUTE_TIME)

f_all = list(FEATURES_RAW)
f_noid = [c for c in f_all if c not in IDENTIFIERS + FINGERPRINTS + ABSOLUTE_TIME]
f_behav = ([c for c in f_noid if c not in RAW_PORTS + ["frame.protocols"]]
           + [c for c in DERIVED if c in df.columns])

REGIMES = {"F-ALL": f_all, "F-NOID": f_noid, "F-BEHAV": f_behav}
save_json(REGIMES, RESULT / "regime_columns.json")
for k, v in REGIMES.items():
    print(f"{k:<8} {len(v):>3} features")
print("\nF-BEHAV:", ", ".join(REGIMES["F-BEHAV"]))

leak = BANNED_BEHAV & set(f_behav)
gate("S4 regimes", not leak and ABS_TIME_COL not in f_behav,
     f"F-BEHAV excludes identifiers/fingerprints/absolute time "
     f"({len(f_behav)} features); leaks={sorted(leak)}")

# ---- per-regime matrices (encoders are fitted inside the fold, Stage 6)
def build_matrix(regime):
    cols = REGIMES[regime]
    cat = [c for c in cols if c in CATEGORICAL]
    num = [c for c in cols if c not in cat]
    Xc = df[cat].astype(str) if cat else pd.DataFrame(index=df.index)
    Xn = (df[num].apply(pd.to_numeric, errors="coerce")
          if num else pd.DataFrame(index=df.index))
    return Xc, Xn, cat, num


MATRICES = {r: build_matrix(r) for r in REGIME_NAMES}
y = df["label"].astype(str).to_numpy()
groups_device = df["device"].to_numpy()
groups_segment = df["segment"].to_numpy()
CLASSES = sorted(pd.unique(y))
ATTACK_CLASSES = [c for c in CLASSES if c != NORMAL]
cls_index = {c: i for i, c in enumerate(CLASSES)}

print("classes:", CLASSES)
print("attack classes:", ATTACK_CLASSES)
for r in REGIME_NAMES:
    _, _, cat, num = MATRICES[r]
    print(f"{r:<8} categorical={len(cat):>2}  numeric={len(num):>2}")


# -------------------------------------------------------------------------
# Stage 5: Evaluation folds
# Construct packet, device and network-segment holdout splits.
# -------------------------------------------------------------------------


from sklearn.model_selection import GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGKF = True
except Exception:
    HAS_SGKF = False

FOLDS, FOLD_META = {}, {}
idx_all = np.arange(len(df))

# ---------------------------------------------------------------- P1
skf = StratifiedKFold(n_splits=N_FOLDS_P1, shuffle=True, random_state=SEED)
FOLDS["P1-random"] = [(tr, te) for tr, te in skf.split(idx_all, y)]
FOLD_META["P1-random"] = "stratified 5-fold over shuffled packets"

# ---------------------------------------------------------------- P2
atk = [d for d in ATTACK_DEVICES if d in set(groups_device)]
ben = [d for d in BENIGN_DEVICES if d in set(groups_device)]
rs = np.random.RandomState(SEED)
ben = list(rs.permutation(ben))

if 2 <= len(atk) <= P2_MAX_FOLDS or not HAS_SGKF:
    n_folds = min(len(atk), P2_MAX_FOLDS)
    atk_folds = [list(a) for a in np.array_split(np.array(atk, dtype=object), n_folds)]
    ben_folds = [list(b) for b in np.array_split(np.array(ben, dtype=object), n_folds)] \
        if ben else [[] for _ in range(n_folds)]
    p2 = []
    for k in range(n_folds):
        held = set(atk_folds[k]) | set(ben_folds[k])
        te = idx_all[np.isin(groups_device, list(held))]
        tr = idx_all[~np.isin(groups_device, list(held))]
        if len(te) and len(tr):
            p2.append((tr, te))
    FOLDS["P2-device"] = p2
    FOLD_META["P2-device"] = (f"leave-one-attacked-device-out ({n_folds} folds); "
                              f"benign devices partitioned round-robin")
else:
    sgkf = StratifiedGroupKFold(n_splits=min(5, max(2, len(atk))),
                                shuffle=True, random_state=SEED)
    FOLDS["P2-device"] = [(tr, te) for tr, te
                          in sgkf.split(idx_all, y, groups=groups_device)]
    FOLD_META["P2-device"] = "StratifiedGroupKFold on device"

# ---------------------------------------------------------------- P3
attacked_segments = sorted({int(s) for s in
                            stats.loc[stats["attack_rows_total"] > 0, "segment"].unique()
                            if s > 0})
p3 = []
for s in attacked_segments:
    te = idx_all[groups_segment == s]
    tr = idx_all[groups_segment != s]
    if len(te) and len(tr) and len(set(y[te])) >= 2:
        p3.append((tr, te))
FOLDS["P3-segment"] = p3
FOLD_META["P3-segment"] = (f"leave-one-segment-out over attacked segments "
                           f"{attacked_segments}; segment {BENIGN_ONLY_SEGMENT} "
                           f"always in train")

print("attacked segments:", attacked_segments)
for k, v in FOLD_META.items():
    print(f"{k:<11} {len(FOLDS[k])} folds -- {v}")

rows, ok_dev, ok_seg2, ok_cls = [], True, True, True
for proto, folds in FOLDS.items():
    for i, (tr, te) in enumerate(folds):
        dtr, dte = set(groups_device[tr]), set(groups_device[te])
        s_tr, s_te = set(groups_segment[tr].tolist()), set(groups_segment[te].tolist())
        cls_te = sorted(set(y[te]))
        atk_te = sorted(d for d in dte if d in set(ATTACK_DEVICES))
        if proto == "P2-device" and (dtr & dte):
            ok_dev = False
        if proto == "P3-segment" and (s_tr & s_te):
            ok_seg2 = False
        if len(cls_te) < 2:
            ok_cls = False
        rows.append({"protocol": proto, "fold": i, "n_train": len(tr), "n_test": len(te),
                     "test_devices": len(dte), "test_attack_devices": ";".join(atk_te),
                     "test_segments": ";".join(str(s) for s in sorted(s_te)),
                     "test_classes": len(cls_te),
                     "test_attack_rows": int((y[te] != NORMAL).sum())})

comp_folds = pd.DataFrame(rows)
comp_folds.to_csv(RESULT / "fold_composition.csv", index=False)
save_json({k: {"n_folds": len(v), "design": FOLD_META[k],
               "sizes": [[int(len(a)), int(len(b))] for a, b in v]}
           for k, v in FOLDS.items()}, RESULT / "folds.json")
print(comp_folds.to_string(index=False))

gate("S5 device disjoint", ok_dev, "no device spans train/test under P2")
gate("S5 segment disjoint", ok_seg2, "no segment spans train/test under P3")
gate("S5 class support", ok_cls, "every test fold carries >= 2 classes")
gate("S5 attack support", bool((comp_folds["test_attack_rows"] > 0).all()),
     "every test fold carries attack traffic")


# -------------------------------------------------------------------------
# Stage 6: Model comparison
# Fit models and preprocessing per training fold; save metrics and matrices.
# -------------------------------------------------------------------------


METRICS_PATH = RESULT / "metrics.csv"
PERCLASS_PATH = RESULT / "per_class.csv"


def load_done():
    if METRICS_PATH.exists():
        m = pd.read_csv(METRICS_PATH)
        if len(m):
            return set(zip(m.protocol, m.regime, m.model, m.fold))
    return set()


def fit_transform_fold(regime, tr, te):
    """Fit every transformer on TRAIN ONLY, then apply to test."""
    Xc, Xn, cat, num = MATRICES[regime]
    parts_tr, parts_te, cat_idx, col = [], [], [], 0
    if cat:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                             encoded_missing_value=-1)
        a = enc.fit_transform(Xc.iloc[tr]) + 1.0       # shift so unknown == 0
        b = enc.transform(Xc.iloc[te]) + 1.0
        # only low-cardinality columns may be declared categorical to HistGB
        for j in range(a.shape[1]):
            if len(np.unique(a[:, j])) <= HGB_MAX_CARD:
                cat_idx.append(col + j)
        parts_tr.append(a); parts_te.append(b); col += a.shape[1]
    if num:
        imp = SimpleImputer(strategy="median")
        a = imp.fit_transform(Xn.iloc[tr]); b = imp.transform(Xn.iloc[te])
        sc = StandardScaler()
        a = sc.fit_transform(a); b = sc.transform(b)
        parts_tr.append(a); parts_te.append(b); col += a.shape[1]
    Xtr = np.hstack(parts_tr).astype("float32")
    Xte = np.hstack(parts_te).astype("float32")
    return np.nan_to_num(Xtr), np.nan_to_num(Xte), cat_idx


def make_model(name, n_features, cat_idx):
    if name == "logreg":
        return LogisticRegression(max_iter=400, class_weight="balanced")
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1,
                                      class_weight="balanced_subsample",
                                      random_state=SEED)
    if name == "histgb":
        mask = None
        if cat_idx:
            mask = np.zeros(n_features, dtype=bool)
            mask[cat_idx] = True
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                      categorical_features=mask,
                                      random_state=SEED,
                                      early_stopping=False)
    raise ValueError(name)


def evaluate(yte, pred):
    p, r, f, s = precision_recall_fscore_support(yte, pred, labels=CLASSES,
                                                 average=None, zero_division=0)
    per = pd.DataFrame({"class": CLASSES, "precision": p, "recall": r,
                        "f1": f, "support": s})
    atk_present = [c for c in ATTACK_CLASSES if c in set(yte)]
    atk_f1 = (f1_score(yte, pred, labels=atk_present, average="macro", zero_division=0)
              if atk_present else np.nan)
    yb, pb = (yte != NORMAL), (pred != NORMAL)
    tp = int((yb & pb).sum()); fn = int((yb & ~pb).sum()); fp = int((~yb & pb).sum())
    return {
        "macro_f1": f1_score(yte, pred, average="macro", zero_division=0),
        "macro_f1_attacks": atk_f1,
        "weighted_f1": f1_score(yte, pred, average="weighted", zero_division=0),
        "balanced_acc": balanced_accuracy_score(yte, pred),
        "mcc": matthews_corrcoef(yte, pred),
        "accuracy": float((pred == yte).mean()),
        "attack_recall": tp / max(1, tp + fn),
        "attack_precision": tp / max(1, tp + fp),
    }, per


print("grid helpers ready")

def torch_mlp_fit_predict(Xtr, ytr_enc, Xte, n_classes, epochs=30, bs=4096, lr=1e-3):
    import torch, torch.nn as nn
    g = torch.Generator().manual_seed(SEED)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(128, 64), nn.ReLU(),
                        nn.Linear(64, n_classes)).to(DEVICE)
    seen = np.unique(ytr_enc)
    cw = np.ones(n_classes, dtype="float32")
    cw[seen] = compute_class_weight("balanced", classes=seen, y=ytr_enc)
    crit = nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float32, device=DEVICE))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt = torch.tensor(Xtr, device=DEVICE)
    yt = torch.tensor(ytr_enc, dtype=torch.long, device=DEVICE)
    n = len(Xt)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n, generator=g).to(DEVICE)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            crit(net(Xt[b]), yt[b]).backward()
            opt.step()
    net.eval()
    outs = []
    with torch.no_grad():
        Xe = torch.tensor(Xte, device=DEVICE)
        for i in range(0, len(Xe), 16384):
            outs.append(net(Xe[i:i + 16384]).argmax(1).cpu().numpy())
    return np.concatenate(outs)


done = load_done()
total = sum(len(v) for v in FOLDS.values()) * len(REGIME_NAMES) * len(MODELS)
print(f"{len(done)} fits already complete, {total} planned")
t_start = time.time()

for proto in PROTOCOLS:
    for regime in REGIME_NAMES:
        for fold_id, (tr, te) in enumerate(FOLDS[proto]):
            tr = np.asarray(tr); te = np.asarray(te)
            todo = [m for m in MODELS if (proto, regime, m, fold_id) not in done]
            if not todo:
                continue
            Xtr, Xte, cat_idx = fit_transform_fold(regime, tr, te)
            ytr, yte = y[tr], y[te]
            for model in todo:
                t0 = time.time()
                try:
                    if model == "mlp":
                        if not HAS_TORCH:
                            print("  skip mlp (torch absent)")
                            continue
                        enc = np.array([cls_index[v] for v in ytr])
                        pred = np.array([CLASSES[i] for i in
                                         torch_mlp_fit_predict(Xtr, enc, Xte,
                                                               len(CLASSES))])
                    else:
                        clf = make_model(model, Xtr.shape[1], cat_idx).fit(Xtr, ytr)
                        pred = clf.predict(Xte)
                except Exception as e:
                    print(f"  !! {proto}/{regime}/{model}/f{fold_id} FAILED: "
                          f"{type(e).__name__}: {e}", flush=True)
                    continue
                dt = time.time() - t0
                sc, per = evaluate(yte, pred)
                row = {"protocol": proto, "regime": regime, "model": model,
                       "fold": fold_id, "n_train": int(len(tr)), "n_test": int(len(te)),
                       "fit_seconds": round(dt, 1),
                       **{k: (round(v, 5) if v == v else np.nan) for k, v in sc.items()}}
                pd.DataFrame([row]).to_csv(METRICS_PATH, mode="a", index=False,
                                           header=not METRICS_PATH.exists())
                per.insert(0, "fold", fold_id); per.insert(0, "model", model)
                per.insert(0, "regime", regime); per.insert(0, "protocol", proto)
                per.to_csv(PERCLASS_PATH, mode="a", index=False,
                           header=not PERCLASS_PATH.exists())
                np.savez_compressed(CONF / f"{proto}_{regime}_{model}_f{fold_id}.npz",
                                    cm=confusion_matrix(yte, pred, labels=CLASSES),
                                    classes=np.array(CLASSES))
                done.add((proto, regime, model, fold_id))
                slow = "   <-- SLOW" if dt > MAX_FIT_MINUTES * 60 else ""
                print(f"{proto:<11} {regime:<7} {model:<7} f{fold_id}  "
                      f"macroF1={sc['macro_f1']:.4f}  atkF1={sc['macro_f1_attacks']:.4f}  "
                      f"acc={sc['accuracy']:.4f}  {dt:.0f}s{slow}", flush=True)
            del Xtr, Xte

print(f"\nStage 6 finished in {(time.time()-t_start)/60:.1f} min")
metrics = pd.read_csv(METRICS_PATH)

expected_keys = {(p, r, m, f) for p in PROTOCOLS for r in REGIME_NAMES
                 for m in MODELS for f in range(len(FOLDS[p]))}
missing = sorted(expected_keys - set(zip(metrics.protocol, metrics.regime,
                                         metrics.model, metrics.fold)))
missing_nonmlp = [k for k in missing if k[2] != "mlp"]
if missing:
    print(f"{len(missing)} fits missing; "
          f"{len(missing) - len(missing_nonmlp)} are MLP "
          f"(torch {'present' if HAS_TORCH else 'ABSENT'})")
    for k in missing_nonmlp[:10]:
        print("   missing:", k)

gate("S6 grid", len(metrics) > 0 and bool(metrics["macro_f1"].notna().all())
     and not missing_nonmlp and (HAS_TORCH or not any(m == "mlp" for m in MODELS)
                                 or len(missing) == len(missing) and not HAS_TORCH),
     f"{len(metrics)}/{total} fits recorded, {len(missing_nonmlp)} non-MLP missing")
if missing and HAS_TORCH:
    raise SystemExit("Some fits did not complete even though torch is available. "
                     "Re-run the model stage and inspect any remaining failures.")


# -------------------------------------------------------------------------
# Stage 7: Federated comparison (optional)
# Evaluate the supplementary federated-learning baseline.
# -------------------------------------------------------------------------


FL_PATH = RESULT / "fl_rounds.csv"


def fedavg(regime, tr, te, rounds=FL_ROUNDS, clients=FL_CLIENTS_PER_ROUND):
    import torch, torch.nn as nn
    Xtr, Xte, _ = fit_transform_fold(regime, tr, te)
    ytr = np.array([cls_index[v] for v in y[tr]])
    yte = y[te]
    devs = groups_device[tr]
    uniq = np.unique(devs)
    parts = {d: np.where(devs == d)[0] for d in uniq}

    def new_net():
        return nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.ReLU(),
                             nn.Linear(256, 128), nn.ReLU(),
                             nn.Linear(128, len(CLASSES))).to(DEVICE)

    glob = new_net()
    seen = np.unique(ytr)
    cw = np.ones(len(CLASSES), dtype="float32")
    cw[seen] = compute_class_weight("balanced", classes=seen, y=ytr)
    crit = nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float32, device=DEVICE))
    Xt = torch.tensor(Xtr, device=DEVICE)
    yt = torch.tensor(ytr, dtype=torch.long, device=DEVICE)
    Xe = torch.tensor(Xte, device=DEVICE)
    rng = np.random.RandomState(SEED)
    hist = []

    for rd in range(1, rounds + 1):
        chosen = rng.choice(uniq, size=min(clients, len(uniq)), replace=False)
        states, sizes = [], []
        for d in chosen:
            idx = parts[d]
            if len(idx) < 16:
                continue
            local = new_net()
            local.load_state_dict(glob.state_dict())
            opt = torch.optim.Adam(local.parameters(), lr=1e-3)
            local.train()
            perm = torch.tensor(rng.permutation(idx), device=DEVICE)
            for i in range(0, len(perm), 2048):
                b = perm[i:i + 2048]
                opt.zero_grad()
                crit(local(Xt[b]), yt[b]).backward()
                opt.step()
            states.append({k: v.detach().clone()
                           for k, v in local.state_dict().items()})
            sizes.append(len(idx))
        if not states:
            continue
        tot = float(sum(sizes))
        glob.load_state_dict({
            k: sum(s[k].float() * (n / tot) for s, n in zip(states, sizes))
                 .to(states[0][k].dtype)
            for k in states[0]})
        glob.eval()
        with torch.no_grad():
            pe = [glob(Xe[i:i + 16384]).argmax(1).cpu().numpy()
                  for i in range(0, len(Xe), 16384)]
        pred = np.array([CLASSES[i] for i in np.concatenate(pe)])
        sc, _ = evaluate(yte, pred)
        hist.append({"regime": regime, "round": rd,
                     "macro_f1": round(sc["macro_f1"], 5),
                     "macro_f1_attacks": round(float(sc["macro_f1_attacks"]), 5)
                     if sc["macro_f1_attacks"] == sc["macro_f1_attacks"] else np.nan,
                     "accuracy": round(sc["accuracy"], 5), "clients": len(states)})
        print(f"  FL {regime:<7} round {rd:>2}: macroF1={hist[-1]['macro_f1']:.4f}",
              flush=True)
    return pd.DataFrame(hist)


if RUN_FEDERATED and HAS_TORCH:
    central = pd.read_csv(METRICS_PATH)
    have = not central[(central.model == "mlp")
                       & (central.protocol == "P2-device")].empty
    gate("S7 baseline present", have,
         "centralised MLP under P2-device exists for comparison")
    all_fl = []
    for regime in ["F-ALL", "F-BEHAV"]:
        for fold_id, (tr, te) in enumerate(FOLDS["P2-device"][:FL_EVAL_FOLDS]):
            h = fedavg(regime, np.asarray(tr), np.asarray(te))
            h["fold"] = fold_id
            all_fl.append(h)
    if all_fl:
        fl = pd.concat(all_fl, ignore_index=True)
        fl.to_csv(FL_PATH, index=False)
        print()
        print(fl.groupby(["round", "regime"])["macro_f1"].mean().unstack().to_string())
else:
    print("Stage 7 skipped (RUN_FEDERATED off, or torch unavailable)")


# -------------------------------------------------------------------------
# Stage 8: Result aggregation
# Create tabular summaries and visual outputs from experiment records.
# -------------------------------------------------------------------------


metrics = pd.read_csv(METRICS_PATH)
perclass = pd.read_csv(PERCLASS_PATH)
probes = pd.read_csv(TABLES / "table2_probes.csv")
hyg = pd.read_csv(RESULT / "hygiene.csv")

agg = (metrics.groupby(["protocol", "regime", "model"])
       .agg(macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
            atk_f1_mean=("macro_f1_attacks", "mean"),
            atk_f1_std=("macro_f1_attacks", "std"),
            acc_mean=("accuracy", "mean"), bacc_mean=("balanced_acc", "mean"),
            mcc_mean=("mcc", "mean"),
            atk_recall_mean=("attack_recall", "mean"),
            folds=("fold", "count"))
       .reset_index().round(4))
agg.to_csv(TABLES / "table3_main.csv", index=False)


def pm(a, b):
    return a.map("{:.3f}".format) + " ± " + b.fillna(0).map("{:.3f}".format)


pretty = agg.copy()
pretty["macro-F1"] = pm(pretty.macro_f1_mean, pretty.macro_f1_std)
pretty["attack macro-F1"] = pm(pretty.atk_f1_mean, pretty.atk_f1_std)

for col, fname in (("macro-F1", "table3_main_wide"),
                   ("attack macro-F1", "table3_attackonly_wide")):
    wide = pretty.pivot_table(index=["regime", "model"], columns="protocol",
                              values=col, aggfunc="first")
    wide = wide.reindex(columns=[p for p in PROTOCOLS if p in wide.columns])
    wide.to_csv(TABLES / f"{fname}.csv")
    try:
        (TABLES / f"{fname}.tex").write_text(wide.to_latex())
    except Exception as e:
        print("latex export skipped:", e)
    print(f"\n=== {col} ===")
    print(wide.to_string())

# ---- headline: the optimism gap ---------------------------------------
def cell(proto, regime, model, col="macro_f1"):
    m = metrics[(metrics.protocol == proto) & (metrics.regime == regime)
                & (metrics.model == model)]
    return m[col].dropna().to_numpy()


pool = metrics[(metrics.protocol == "P2-device") & (metrics.regime == "F-BEHAV")]
best_model = (pool.groupby("model")["macro_f1"].mean().idxmax()
              if len(pool) else metrics.groupby("model")["macro_f1"].mean().idxmax())

opt_hi = cell("P1-random", "F-ALL", best_model)
opt_lo = cell("P2-device", "F-BEHAV", best_model)
gap = float(np.mean(opt_hi) - np.mean(opt_lo)) if len(opt_hi) and len(opt_lo) else np.nan

if len(opt_hi) and len(opt_lo):
    rs = np.random.RandomState(SEED)
    boot = np.array([rs.choice(opt_hi, len(opt_hi), replace=True).mean()
                     - rs.choice(opt_lo, len(opt_lo), replace=True).mean()
                     for _ in range(5000)])
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
else:
    ci = [np.nan, np.nan]

headline = {
    "best_model_deployable": str(best_model),
    "macroF1_P1_FALL": round(float(np.mean(opt_hi)), 4) if len(opt_hi) else None,
    "macroF1_P2_FBEHAV": round(float(np.mean(opt_lo)), 4) if len(opt_lo) else None,
    "delta_optimism": round(gap, 4) if gap == gap else None,
    "delta_ci95": [round(c, 4) for c in ci] if ci[0] == ci[0] else None,
    "identifier_probe_macroF1": float(probes.loc[0, "macroF1"]),
    "time_probe_macroF1": float(probes.loc[1, "macroF1"]),
    "exact_duplicate_rate": float(hyg.loc[0, "exact_duplicate_rate"]),
    "cross_label_duplicate_rate": float(hyg.loc[0, "cross_label_duplicate_group_rate"]),
    "n_attack_devices": len(ATTACK_DEVICES),
    "n_devices": int(df["device"].nunique()),
    "classes": CLASSES, "scale": SCALE, "rows": int(len(df)),
    "config_hash": CONFIG_HASH, "p2_design": FOLD_META.get("P2-device"),
}
save_json(headline, RESULT / "headline.json")
print(json.dumps(headline, indent=2))

# ---- Fig. 2: the optimism gap, made visual ----------------------------
sub = agg[agg.model == best_model]
protos = [p for p in PROTOCOLS if p in set(sub.protocol)]
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (vcol, ecol, title) in zip(axes, [
        ("macro_f1_mean", "macro_f1_std", "All classes"),
        ("atk_f1_mean", "atk_f1_std", "Attack classes only")]):
    xs = np.arange(len(REGIME_NAMES))
    w = 0.8 / max(1, len(protos))
    for i, proto in enumerate(protos):
        vals = [sub.loc[(sub.protocol == proto) & (sub.regime == r), vcol].mean()
                for r in REGIME_NAMES]
        errs = [sub.loc[(sub.protocol == proto) & (sub.regime == r), ecol].mean()
                for r in REGIME_NAMES]
        errs_clean = np.nan_to_num(errs)
        bar_x = xs + (i - (len(protos) - 1) / 2) * w
        ax.bar(bar_x, vals, w, yerr=errs_clean, capsize=3, label=proto)
        for x, v, e in zip(bar_x, vals, errs_clean):
            if v == v:  # skip NaN
                ax.text(x, v + e + 0.02, f"{v:.2f}", ha="center",
                        va="bottom", fontsize=6, rotation=90)
    ax.set_xticks(xs); ax.set_xticklabels(REGIME_NAMES)
    ax.set_ylim(0, 1); ax.set_title(title)
axes[0].set_ylabel("macro-F1")
axes[0].legend(fontsize=8)
fig.suptitle(f"Optimism gap on Gotham 2025 ({best_model})")
fig.tight_layout()
fig.savefig(FIGS / "fig2_optimism_gap.png", dpi=600, bbox_inches="tight")
fig.savefig(FIGS / "fig2_optimism_gap.pdf", dpi=600, bbox_inches="tight")
plt.close(fig)

# ---- Fig. 3: confusion matrix, deployable configuration ---------------
cms = [np.load(p)["cm"] for p in
       sorted(CONF.glob(f"P2-device_F-BEHAV_{best_model}_f*.npz"))]
if cms:
    cm = np.sum(cms, axis=0).astype(float)
    cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(cmn, vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES, fontsize=8)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="w" if cmn[i, j] < 0.6 else "k")
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"F-BEHAV · device-disjoint · {best_model}")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_confusion.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGS / "fig3_confusion.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Fig. 3 written")
else:
    print(f"*** Fig. 3 SKIPPED: no confusion-matrix files found for "
          f"best_model='{best_model}' matching "
          f"'P2-device_F-BEHAV_{best_model}_f*.npz' in {CONF}. "
          f"This usually means that model's P2-device/F-BEHAV folds "
          f"haven't completed yet in Stage 6, or best_model changed to "
          f"a model whose folds are still missing. Check Stage 6's "
          f"metrics.csv for that model/protocol/regime combination.")

# ---- Fig. 4: federated vs centralised ---------------------------------
if FL_PATH.exists():
    fl = pd.read_csv(FL_PATH)
    fig, ax = plt.subplots(figsize=(6.2, 4))
    for k, (regime, g) in enumerate(fl.groupby("regime")):
        m = g.groupby("round")["macro_f1"].mean()
        ax.plot(m.index, m.values, marker="o", label=f"FedAvg {regime}")
        for x, v in zip(m.index, m.values):
            ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points",
                        xytext=(0, 8 if k == 0 else -12), ha="center", fontsize=6)
    for regime in ["F-ALL", "F-BEHAV"]:
        c = metrics[(metrics.model == "mlp") & (metrics.protocol == "P2-device")
                    & (metrics.regime == regime)]["macro_f1"].mean()
        if c == c:
            ax.axhline(c, linestyle="--", linewidth=1,
                       label=f"central MLP {regime}")
            ax.annotate(f"{c:.2f}", (fl["round"].max(), c),
                        textcoords="offset points", xytext=(6, 0),
                        ha="left", va="center", fontsize=6)
    ax.set_xlabel("federated round"); ax.set_ylabel("macro-F1"); ax.set_ylim(0, 1)
    ax.set_title("Federated vs centralised, device-disjoint")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_federated.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGS / "fig4_federated.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Fig. 4 written")

print("\nfigures ->", FIGS)


# -------------------------------------------------------------------------
# Stage 9: Output packaging
# Write the run report and package experiment outputs.
# -------------------------------------------------------------------------


report = [
    "# Gotham 2025 IDS evaluation: execution report", "",
    f"- run tag: `{RUN_TAG}`",
    f"- scale: **{SCALE}** ({ROWS_PER_DEVICE} rows/device), labels: {LABEL_GRANULARITY}",
    f"- working table: {len(df):,} rows x {df.shape[1]} cols, "
    f"{df['device'].nunique()} devices, {len(ATTACK_DEVICES)} attack-bearing",
    f"- classes: {', '.join(CLASSES)}",
    f"- fits recorded: {len(metrics)}",
    f"- total fit time: {metrics['fit_seconds'].sum()/60:.1f} min",
    "", "## Fold designs",
]
for k, v in FOLD_META.items():
    report.append(f"- **{k}** ({len(FOLDS[k])} folds): {v}")
report += ["", "## Headline", "```", json.dumps(headline, indent=2), "```",
           "", "## Attack-bearing devices", "```",
           "\n".join(ATTACK_DEVICES), "```",
           "", "## Environment", "```",
           (RESULT / "environment.txt").read_text(), "```"]
if UNKNOWN_LABELS:
    report += ["", "## Unrecognised label strings", "```",
               "\n".join(sorted(UNKNOWN_LABELS)), "```"]
(RESULT / "RUN_REPORT.md").write_text("\n".join(report))

zip_path = ROOT / f"ICIC2026_gotham_results_{SCALE}.zip"
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(RESULT.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(RESULT))
print("wrote", zip_path, f"({zip_path.stat().st_size/1e6:.1f} MB)")

try:
    from google.colab import files
    files.download(str(zip_path))
except Exception:
    print("Download it from the file browser on the left, or from Drive.")

gate("S9 package", zip_path.exists(), str(zip_path))
print("\nResults archive is ready.")