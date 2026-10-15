| arm (fix removed) | a2av | dxe | fsdp | lbar | pp | rkv | tp | ua2a | pooled | direction |
|---|---|---|---|---|---|---|---|---|---|---|
| full (v16b) | 0.9 | 0.73 | 0.81 | 0.9 | 0.89 | 0.75 | 0.9 | 0.73 | 0.69 | 31/33 |
| − real harness shapes (toy shapes × multiplier) | 0.9 | 0.73 | 0.52 | 0.9 | 0.89 | 0.75 | 0.74 | 0.73 | 0.62 | 34/39 |
| − bytes actually moved (input bytes only) | 0.9 | 0.27 | 0.91 | 0.9 | 0.89 | 0.93 | 0.9 | 0.73 | 0.56 | 30/35 |
| − volume-charged local ops (flat per-op cost) | 0.9 | 0.27 | 0.81 | 0.9 | 0.89 | 0.75 | 0.9 | 0.73 | 0.53 | 28/33 |
| − measured collective curves (topology constants) | 0.29 | 0.73 | -0.41 | -0.2 | -0.8 | 0.75 | -0.26 | 0.73 | 0.27 | 16/21 |
| − fusion-aware dispatch runs (any local op resets) | 0.9 | 0.02 | 0.81 | 0.9 | 0.89 | 0.75 | 0.9 | 0.73 | 0.57 | 29/33 |
| − XLA concat-of-slices rule | 0.9 | 0.73 | 0.81 | 0.9 | 0.89 | 0.75 | 0.9 | 0.73 | 0.69 | 31/33 |
| − XLA dead-collective elimination | 0.9 | 0.73 | 0.81 | 0.9 | 0.89 | 0.58 | 0.9 | 0.73 | 0.64 | 30/33 |
| − forward+backward factor on volume terms | 0.9 | 0.56 | 0.81 | 0.9 | 0.89 | 0.75 | 0.9 | 0.73 | 0.64 | 28/30 |
| − scoring helpers by the code they run | 0.56 | 0.73 | 0.81 | 0.9 | 0.89 | 0.75 | 0.9 | 0.73 | 0.65 | 30/33 |
