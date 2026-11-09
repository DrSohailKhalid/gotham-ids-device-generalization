# ICIC 2026 - Gotham 2025 run report

- run tag: `main_coarse6_e21113edae`
- scale: **main** (8000 rows/device), labels: coarse6
- working table: 404,324 rows x 49 cols, 78 devices, 8 attack-bearing
- classes: Brute Force, C&C Communication, DoS, Infection, Network Scanning, Normal, Unknown
- fits recorded: 192
- total fit time: 274.6 min

## Fold designs
- **P1-random** (5 folds): stratified 5-fold over shuffled packets
- **P2-device** (8 folds): leave-one-attacked-device-out (8 folds); benign devices partitioned round-robin
- **P3-segment** (3 folds): leave-one-segment-out over attacked segments [1, 2, 4]; segment 3 always in train

## Headline
```
{
  "best_model_deployable": "rf",
  "macroF1_P1_FALL": 0.9984,
  "macroF1_P2_FBEHAV": 0.7843,
  "delta_optimism": 0.2141,
  "delta_ci95": [
    0.1599,
    0.27
  ],
  "identifier_probe_macroF1": 0.9927,
  "time_probe_macroF1": 0.7087,
  "exact_duplicate_rate": 0.0299,
  "cross_label_duplicate_rate": 0.0,
  "n_attack_devices": 8,
  "n_devices": 78,
  "classes": [
    "Brute Force",
    "C&C Communication",
    "DoS",
    "Infection",
    "Network Scanning",
    "Normal",
    "Unknown"
  ],
  "scale": "main",
  "rows": 404324,
  "config_hash": "e21113edae",
  "p2_design": "leave-one-attacked-device-out (8 folds); benign devices partitioned round-robin"
}
```

## Attack-bearing devices
```
air-quality-1
building-monitor-1
city-power-1
combined-cycle-1
combined-cycle-10
domotic-monitor-1
ip-camera-museum-1
ip-camera-street-1
```

## Environment
```
python        3.13.15
numpy         2.1.3
pandas        2.2.3
scikit-learn  1.6.1
torch         2.11.0+cu128
device        cuda
seed          20260101
scale         main (8000 rows/device, devices=all)
label_mode    coarse6
config_hash   e21113edae
run_tag       main_coarse6_e21113edae
```