# Step 0 — compile-cost premise check

manifest modules: 529; bench rows: default 90, cold 24, flags 49, versions 48

## 1. Compile cost by family (default flags, installed compiler, repeat 0)

| source | family | n | PASS | FAIL | TIMEOUT | OOM | wall median s | p90 | max | RSS median GB | RSS max GB | NEFF median MB | NEFF max MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| collective | collective | 3 | 3 | 0 | 0 | 0 | 2.0 | 2.7 | 2.7 | 0.24 | 0.24 | 0.02 | 0.03 |
| mutation | synth_attn | 2 | 2 | 0 | 0 | 0 | 141.1 | 220.6 | 220.6 | 4.71 | 7.98 | 13.46 | 22.13 |
| mutation | synth_conv | 1 | 1 | 0 | 0 | 0 | 143.1 | 143.1 | 143.1 | 3.11 | 3.11 | 2.52 | 2.52 |
| mutation | synth_mlp | 1 | 1 | 0 | 0 | 0 | 6.4 | 6.4 | 6.4 | 0.27 | 0.27 | 0.42 | 0.42 |
| real | hf | 3 | 3 | 0 | 0 | 0 | 54.3 | 223.8 | 223.8 | 1.18 | 5.59 | 3.98 | 10.79 |
| synth | synth_attn | 1 | 1 | 0 | 0 | 0 | 6.8 | 6.8 | 6.8 | 0.27 | 0.27 | 0.16 | 0.16 |
| synth | synth_conv | 2 | 2 | 0 | 0 | 0 | 40.1 | 75.9 | 75.9 | 1.24 | 2.20 | 3.21 | 6.24 |
| synth | synth_elementwise | 3 | 3 | 0 | 0 | 0 | 2.8 | 87.3 | 87.3 | 0.25 | 2.78 | 0.15 | 8.72 |
| synth | synth_gather | 3 | 3 | 0 | 0 | 0 | 2.0 | 175.7 | 175.7 | 0.24 | 10.63 | 0.02 | 9.99 |
| synth | synth_many_small | 3 | 3 | 0 | 0 | 0 | 7.8 | 166.8 | 166.8 | 0.28 | 0.57 | 0.06 | 0.41 |
| synth | synth_mlp | 2 | 2 | 0 | 0 | 0 | 12.7 | 22.9 | 22.9 | 0.38 | 0.51 | 0.87 | 1.70 |
| synth | synth_reduce | 3 | 3 | 0 | 0 | 0 | 3.1 | 4.0 | 4.0 | 0.25 | 0.26 | 0.23 | 1.15 |
| synth | synth_sort | 3 | 3 | 0 | 0 | 0 | 4.1 | 54.7 | 54.7 | 0.26 | 3.19 | 3.27 | 13.77 |

All PASS modules: n=30, wall min 2.0 s, median 5.3, p90 175.7, p99 223.8, max 223.8; total compile time 22.6 min; share > 60 s: 26.7%; > 300 s: 0.0%; max/min ratio 114×

### 1b. Framework-side compile time at capture (every step graph compiled once by the torch-xla wrapper)

| source | family | n | min s | median | p90 | max |
|---|---|---|---|---|---|---|
| collective | collective | 10 | 1.9 | 2.0 | 2.6 | 2.6 |
| mutation | hf | 26 | 21.4 | 61.4 | 104.5 | 185.4 |
| mutation | synth_attn | 16 | 63.2 | 147.7 | 287.7 | 290.9 |
| mutation | synth_conv | 5 | 140.2 | 160.9 | 308.1 | 308.1 |
| mutation | synth_mlp | 17 | 4.7 | 7.0 | 12.2 | 15.0 |
| real | hf | 25 | 10.3 | 43.6 | 130.6 | 229.1 |
| synth | synth_attn | 16 | 6.5 | 21.3 | 73.1 | 117.4 |
| synth | synth_conv | 8 | 4.7 | 64.9 | 161.4 | 161.4 |
| synth | synth_elementwise | 8 | 2.6 | 5.8 | 90.3 | 90.3 |
| synth | synth_gather | 22 | 2.1 | 2.2 | 8.5 | 194.8 |
| synth | synth_many_small | 19 | 2.8 | 9.3 | 172.3 | 265.0 |
| synth | synth_mlp | 24 | 2.4 | 4.2 | 13.6 | 22.6 |
| synth | synth_reduce | 6 | 2.6 | 3.3 | 4.3 | 4.3 |
| synth | synth_sort | 12 | 2.0 | 4.7 | 56.9 | 59.7 |

All step graphs n=214: median 11.9 s, p90 140.2, max 308.1; > 60 s: 28%; > 120 s: 12%; max/min 164×

## 2. Noise: warm repeats, cold cache, bench vs framework wrapper

- warm repeats: 30 modules with ≥2 repeats; CV median 0.007, p90 0.024, max 0.028; max/min ratio median 1.016, max 1.06
  - CV for compiles > 10 s: median 0.007 (n=12); ≤ 10 s: 0.008 (n=18)
- cold page cache vs warm repeat 0: 24 modules; ratio median 1.535, p90 2.877, max 2.921; absolute extra median 3.88 s
- framework-wrapper CompileTime vs standalone CLI wall (single-step-graph programs, n=30): ratio median 1.038, p10 0.966, p90 1.113

## 3a. Compiler flags (subset; ratio to default flags, same module)

| arm | n | PASS | FAIL/TIMEOUT/OOM | wall ratio to default: median | p10 | p90 | max | RSS ratio median | NEFF ratio median | status changes |
|---|---|---|---|---|---|---|---|---|---|---|
| O1 | 24 | 24 | 0/0/0 | 0.998 | 0.324 | 1.016 | 4.67 | 1.000 | 1.000 | 0 |
| O3 | 24 | 24 | 0/0/0 | 0.995 | 0.959 | 1.011 | 1.02 | 1.000 | 1.000 | 0 |
| transformer | 1 | 1 | 0/0/0 | 0.995 | 0.995 | 0.995 | 0.99 | 1.001 | 1.000 | 0 |

## 3b. Compiler versions (subset; ratio to installed 2.21, default flags)

| arm | n | PASS | FAIL/TIMEOUT/OOM | wall ratio to default: median | p10 | p90 | max | RSS ratio median | NEFF ratio median | status changes |
|---|---|---|---|---|---|---|---|---|---|---|
| cc2.19.8089.0 | 24 | 24 | 0/0/0 | 0.958 | 0.815 | 1.157 | 10.90 | 0.954 | 0.879 | 0 |
| cc2.24.8799.0 | 24 | 24 | 0/0/0 | 0.643 | 0.292 | 0.872 | 29.48 | 1.028 | 1.000 | 0 |

### 3c. Per-module arms that change compile time by more than 1.5× (either direction)

| arm | module | default s | arm s | ratio | arm RSS GB | top passes in arm (s) |
|---|---|---|---|---|---|---|
| cc2.24.8799.0 | collective__kindall_gather_numel4096_count1__M7429069748 | 2.0 | 58.4 | 29.48 | 0.23 | PGLayoutTilingPipeline 0, PAGLayoutOpt 0, ParAxesAnnotation 0 |
| cc2.19.8089.0 | collective__kindall_gather_numel4096_count1__M7429069748 | 2.0 | 21.6 | 10.90 | 0.21 | SFKVectorizer 0, PGLayoutTilingPipeline 0, PGTiling 0 |
| cc2.24.8799.0 | mut__synth_conv__ch128_layers8_img64_batch8_train1__dtype_ | 143.1 | 1071.1 | 7.49 | 23.45 | PGLayoutTilingPipeline 2, TensorInitialization 2, BirCodeGenLoop 1 |
| O1 | hf__archgpt2_layers12_hidden768_heads12_batch2_seq512_trai | 223.8 | 31.6 | 0.14 | 1.46 | PGLayoutTilingPipeline 9, PGTiling 3, AliasDependencyReset 3 |
| O1 | mut__synth_conv__ch128_layers8_img64_batch8_train1__dtype_ | 143.1 | 668.8 | 4.67 | 6.56 | PGLayoutTilingPipeline 22, PGTiling 7, InlineNativeKernels 5 |
| cc2.24.8799.0 | synth_conv__ch32_layers8_img64_batch8_train1__M6974560977 | 75.9 | 17.0 | 0.22 | 0.44 | PGLayoutTilingPipeline 2, BirCodeGenLoop 1, TensorInitialization 1 |
| cc2.24.8799.0 | synth_elementwise__n_ops512_numel100000_seed1__M5020559584 | 87.3 | 22.2 | 0.25 | 0.40 | PGLayoutTilingPipeline 8, InsertIOTransposes 8, NeuronSimplifyPredicates 3 |
| cc2.24.8799.0 | mut__synth_attn__dim1024_heads8_layers4_seq1024_batch4_tra | 220.6 | 64.3 | 0.29 | 4.33 | PGLayoutTilingPipeline 2, SFKVectorizer 2, TritiumFusion 1 |
| O1 | mut__synth_attn__dim1024_heads8_layers4_seq1024_batch4_tra | 220.6 | 69.8 | 0.32 | 5.34 | PGLayoutTilingPipeline 6, SFKVectorizer 2, PGTiling 2 |
| O1 | mut__synth_attn__dim1024_heads8_layers4_seq1024_batch4_tra | 61.5 | 20.0 | 0.32 | 1.16 | PGLayoutTilingPipeline 4, SFKVectorizer 2, PGTiling 2 |
| cc2.24.8799.0 | collective__kindstacked_ar_numel1048576_count8__M402368112 | 2.7 | 1.4 | 0.54 | 0.25 | PGLayoutTilingPipeline 0, PAGLayoutOpt 0, InferNonlocalTensors 0 |
| cc2.24.8799.0 | synth_many_small__n_mats1024_mat64_modeunrolled__M49463906 | 166.8 | 90.8 | 0.54 | 0.61 | PGLayoutTilingPipeline 25, PGTiling 17, AGOrderingAnalysisPass 13 |
| cc2.24.8799.0 | synth_gather__kindindex_add_vocab1024_dim256_n_idx256__M01 | 2.0 | 1.2 | 0.57 | 0.25 | PGLayoutTilingPipeline 0, PAGLayoutOpt 0, ParAxesAnnotation 0 |
| cc2.24.8799.0 | synth_gather__kindindex_add_vocab1024_dim256_n_idx8192__M5 | 2.0 | 1.2 | 0.58 | 0.25 | PGLayoutTilingPipeline 0, ParAxesAnnotation 0, PAGLayoutOpt 0 |
| cc2.24.8799.0 | synth_gather__kindgather_vocab1024_dim256_n_idx8192__M9874 | 175.7 | 104.8 | 0.60 | 10.77 | BirCodeGenLoop 0, PGLayoutTilingPipeline 0, ParAxesAnnotation 0 |
| cc2.24.8799.0 | synth_elementwise__n_ops8_numel100000_seed0__M2539830541 | 2.6 | 1.6 | 0.61 | 0.25 | PGLayoutTilingPipeline 0, SplitAPUnionSets 0, NeuronSimplifyPredicates 0 |
| cc2.24.8799.0 | synth_many_small__n_mats16_mat64_modedistinct_shapes__M509 | 2.7 | 1.7 | 0.62 | 0.25 | PGLayoutTilingPipeline 0, DataLocalityOpt 0, PGTiling 0 |
| cc2.24.8799.0 | synth_many_small__n_mats128_mat64_modebatched__M0438828757 | 7.8 | 4.9 | 0.62 | 0.29 | PGLayoutTilingPipeline 2, PGTiling 1, AGOrderingAnalysisPass 0 |
| cc2.24.8799.0 | synth_elementwise__n_ops8_numel10000000_seed1__M7116628037 | 2.8 | 1.7 | 0.63 | 0.26 | PGLayoutTilingPipeline 0, NeuronSimplifyPredicates 0, SplitAPUnionSets 0 |
| cc2.24.8799.0 | hf__archgpt2_layers12_hidden768_heads12_batch2_seq512_trai | 223.8 | 146.4 | 0.65 | 4.30 | PGLayoutTilingPipeline 14, SFKVectorizer 7, PGTiling 5 |
| cc2.24.8799.0 | hf__archgpt2_layers2_hidden768_heads12_batch2_seq512_train | 54.3 | 35.8 | 0.66 | 1.10 | PGLayoutTilingPipeline 3, SFKVectorizer 1, PGTiling 1 |
| cc2.24.8799.0 | hf__archllama_layers1_hidden512_heads8_batch1_seq512_train | 10.1 | 6.7 | 0.66 | 0.32 | BirCodeGenLoop 1, PGLayoutTilingPipeline 1, DramToDramTranspose 0 |

## 4. Near-identical pairs (one parameter changed)

(compile times: 30 programs from the bench, 178 from capture-time framework compiles)

| parent | mutation | parent wall s | child wall s | ratio |
|---|---|---|---|---|
| hf__archllama_layers4_hidden512_heads8_batch1_seq512_tr | batch_half | 42.1 | 165.9 | 3.94 |
| hf__archgpt2_layers2_hidden768_heads12_batch2_seq512_tr | batch_half | 54.3 | 185.4 | 3.42 |
| hf__archgpt2_layers2_hidden768_heads12_batch2_seq512_tr | train_off | 54.3 | 21.4 | 0.39 |
| synth_attn__dim1024_heads8_layers4_seq1024_batch4_train | dim_x2 | 117.4 | 290.9 | 2.48 |
| synth_attn__dim1024_heads8_layers4_seq1024_batch4_train | seq_x2 | 117.4 | 287.7 | 2.45 |
| synth_mlp__d512_hidden2048_depth4_batch1024_actgelu_tra | hidden_x2 | 6.7 | 15.0 | 2.26 |
| synth_attn__dim1024_heads8_layers4_seq1024_batch4_train | batch_x2 | 117.4 | 229.5 | 1.95 |
| hf__archgpt2_layers2_hidden768_heads12_batch2_seq512_tr | hidden_x2 | 54.3 | 104.5 | 1.92 |
| synth_conv__ch128_layers8_img64_batch8_train1 | batch_x2 | 161.4 | 308.1 | 1.91 |
| synth_attn__dim1024_heads8_layers4_seq1024_batch4_train | seq_half | 117.4 | 61.5 | 0.52 |
| synth_attn__dim1024_heads8_layers4_seq1024_batch4_train | batch_half | 117.4 | 220.6 | 1.88 |
| synth_attn__dim1024_heads8_layers4_seq1024_batch4_train | train_off | 117.4 | 63.2 | 0.54 |
| synth_mlp__d512_hidden2048_depth4_batch1024_actgelu_tra | hidden_plus1 | 6.7 | 12.2 | 1.84 |
| hf__archgpt2_layers2_hidden768_heads12_batch2_seq512_tr | seq_x2 | 54.3 | 98.4 | 1.81 |
| synth_attn__dim1024_heads8_layers4_seq1024_batch4_train | seq_plus1 | 117.4 | 212.1 | 1.81 |
| hf__archgpt2_layers2_hidden768_heads12_batch2_seq512_tr | seq_plus1 | 54.3 | 89.9 | 1.66 |
| synth_mlp__d512_hidden2048_depth4_batch1024_actgelu_tra | hidden_minus1 | 6.7 | 10.9 | 1.63 |
| hf__archllama_layers4_hidden512_heads8_batch1_seq512_tr | seq_plus1 | 42.1 | 67.0 | 1.59 |
| hf__archgpt2_layers2_hidden768_heads12_batch2_seq512_tr | batch_minus1 | 54.3 | 34.1 | 0.63 |
| synth_attn__dim1024_heads8_layers4_seq1024_batch4_train | dim_half | 117.4 | 74.2 | 0.63 |
| hf__archgpt2_layers2_hidden768_heads12_batch2_seq512_tr | batch_x2 | 54.3 | 84.5 | 1.56 |
| hf__archllama_layers4_hidden512_heads8_batch1_seq512_tr | train_off | 42.1 | 27.6 | 0.66 |
| hf__archllama_layers4_hidden512_heads8_batch1_seq512_tr | seq_x2 | 42.1 | 62.6 | 1.49 |
| hf__archllama_layers4_hidden512_heads8_batch1_seq512_tr | batch_plus1 | 42.1 | 61.9 | 1.47 |
| hf__archllama_layers4_hidden512_heads8_batch1_seq512_tr | batch_x2 | 42.1 | 60.9 | 1.45 |
| synth_attn__dim1024_heads8_layers4_seq1024_batch4_train | layers_plus1 | 117.4 | 167.1 | 1.42 |
| synth_mlp__d512_hidden2048_depth4_batch1024_actgelu_tra | train_off | 6.7 | 4.7 | 0.70 |
| hf__archgpt2_layers2_hidden768_heads12_batch2_seq512_tr | seq_half | 54.3 | 38.3 | 0.71 |
| synth_mlp__d512_hidden2048_depth4_batch1024_actgelu_tra | batch_x2 | 6.7 | 9.2 | 1.39 |
| hf__archllama_layers4_hidden512_heads8_batch1_seq512_tr | hidden_x2 | 42.1 | 57.8 | 1.37 |
| hf__archgpt2_layers2_hidden768_heads12_batch2_seq512_tr | hidden_half | 54.3 | 39.8 | 0.73 |
| synth_mlp__d512_hidden2048_depth4_batch1024_actgelu_tra | batch_plus1 | 6.7 | 8.9 | 1.34 |
| hf__archgpt2_layers2_hidden768_heads12_batch2_seq512_tr | layers_plus1 | 54.3 | 72.4 | 1.33 |
| hf__archgpt2_layers2_hidden768_heads12_batch2_seq512_tr | batch_plus1 | 54.3 | 71.3 | 1.31 |
| synth_mlp__d512_hidden2048_depth4_batch1024_actgelu_tra | hidden_half | 6.7 | 5.2 | 0.78 |
| synth_attn__dim1024_heads8_layers4_seq1024_batch4_train | batch_plus1 | 117.4 | 149.3 | 1.27 |
| hf__archllama_layers4_hidden512_heads8_batch1_seq512_tr | seq_minus1 | 42.1 | 53.0 | 1.26 |
| hf__archllama_layers4_hidden512_heads8_batch1_seq512_tr | layers_plus1 | 42.1 | 52.8 | 1.25 |
| synth_attn__dim1024_heads8_layers4_seq1024_batch4_train | seq_minus1 | 117.4 | 146.1 | 1.24 |
| synth_attn__dim1024_heads8_layers4_seq1024_batch4_train | batch_minus1 | 117.4 | 94.6 | 0.81 |
| synth_attn__dim1024_heads8_layers4_seq1024_batch4_train | causal_off | 117.4 | 145.0 | 1.24 |
| synth_mlp__d512_hidden2048_depth4_batch1024_actgelu_tra | batch_half | 6.7 | 5.4 | 0.82 |
| synth_mlp__d512_hidden2048_depth4_batch1024_actgelu_tra | depth_plus1 | 6.7 | 7.9 | 1.18 |
| hf__archgpt2_layers2_hidden768_heads12_batch2_seq512_tr | seq_minus1 | 54.3 | 64.1 | 1.18 |
| synth_conv__ch128_layers8_img64_batch8_train1 | batch_minus1 | 161.4 | 140.2 | 0.87 |
| synth_mlp__d512_hidden2048_depth4_batch1024_actgelu_tra | batch_minus1 | 6.7 | 7.6 | 1.14 |
| synth_conv__ch128_layers8_img64_batch8_train1 | dtype_bf16 | 161.4 | 143.1 | 0.89 |
| synth_mlp__d512_hidden2048_depth4_batch1024_actgelu_tra | d_plus1 | 6.7 | 7.4 | 1.11 |
| synth_attn__dim1024_heads8_layers4_seq1024_batch4_train | norm_off | 117.4 | 108.1 | 0.92 |
| synth_mlp__d512_hidden2048_depth4_batch1024_actgelu_tra | dtype_bf16 | 6.7 | 6.2 | 0.93 |
| hf__archllama_layers4_hidden512_heads8_batch1_seq512_tr | seq_half | 42.1 | 39.4 | 0.94 |
| hf__archllama_layers4_hidden512_heads8_batch1_seq512_tr | hidden_half | 42.1 | 40.1 | 0.95 |
| synth_mlp__d512_hidden2048_depth4_batch1024_actgelu_tra | d_minus1 | 6.7 | 7.0 | 1.05 |
| synth_mlp__d512_hidden2048_depth4_batch1024_actgelu_tra | d_half | 6.7 | 6.4 | 0.97 |
| hf__archgpt2_layers2_hidden768_heads12_batch2_seq512_tr | dtype_bf16 | 54.3 | 52.6 | 0.97 |
| synth_mlp__d512_hidden2048_depth4_batch1024_actgelu_tra | d_x2 | 6.7 | 6.9 | 1.03 |
| synth_mlp__d512_hidden2048_depth4_batch1024_actgelu_tra | act_none | 6.7 | 6.5 | 0.98 |
| synth_conv__ch128_layers8_img64_batch8_train1 | batch_half | 161.4 | 159.5 | 0.99 |
| synth_attn__dim1024_heads8_layers4_seq1024_batch4_train | dtype_bf16 | 117.4 | 118.3 | 1.01 |
| hf__archllama_layers4_hidden512_heads8_batch1_seq512_tr | dtype_bf16 | 42.1 | 42.3 | 1.00 |
| synth_conv__ch128_layers8_img64_batch8_train1 | layers_plus1 | 161.4 | 160.9 | 1.00 |
| synth_mlp__d512_hidden2048_depth4_batch1024_actgelu_tra | act_relu | 6.7 | 6.7 | 1.00 |

62 pairs; |log ratio| median 0.301; pairs with >20% change: 42; >2×: 6
±1 dimension changes: 23 pairs, ratio median 1.259, max 1.84, min 0.63

## 5. Cheap-feature baselines (default flags, repeat 0)

n=30 PASS modules with features; targets: log(compile wall s), log(peak RSS)

| target | model | CV | n | R² | MAPE | median APE | Spearman | within 2× |
|---|---|---|---|---|---|---|---|---|
| wall | ridge | random 5-fold | 30 | 0.560 | 1.12 | 0.64 | 0.837 | 0.57 |
| wall | ridge | leave-one-family-out | 30 | 0.450 | 0.95 | 0.77 | 0.722 | 0.43 |
| wall | gbdt | random 5-fold | 30 | 0.537 | 0.69 | 0.39 | 0.807 | 0.70 |
| wall | gbdt | leave-one-family-out | 30 | 0.582 | 0.53 | 0.44 | 0.805 | 0.70 |
| rss | ridge | random 5-fold | 30 | 0.452 | 0.90 | 0.57 | 0.665 | 0.60 |
| rss | ridge | leave-one-family-out | 30 | 0.295 | 0.84 | 0.65 | 0.539 | 0.50 |
| rss | gbdt | random 5-fold | 30 | 0.360 | 0.72 | 0.09 | 0.811 | 0.73 |
| rss | gbdt | leave-one-family-out | 30 | 0.462 | 0.37 | 0.15 | 0.778 | 0.73 |

GBDT top features for log(wall): n_instructions 0.22, proto_bytes 0.15, n_entry_instructions 0.14, n_broadcast 0.11, total_tensor_bytes 0.07, elementwise_bytes 0.07, frac_f32 0.07, largest_tensor_bytes 0.05, n_dtypes 0.04, output_bytes 0.02, n_params 0.01, n_iota 0.01
- Spearman(log wall, n_instructions) = 0.821
- Spearman(log wall, proto_bytes) = 0.813
- Spearman(log wall, total_tensor_bytes) = 0.752
- Spearman(log wall, dot_flops) = 0.552

### 5b. Same baselines on capture-time compile times of all step graphs (n=214, one compile each)

| model | CV | n | R² | MAPE | median APE | Spearman | within 2× |
|---|---|---|---|---|---|---|---|
| ridge | random 5-fold | 214 | 0.775 | 0.69 | 0.37 | 0.885 | 0.75 |
| ridge | leave-one-family-out | 214 | -0.013 | 1.16 | 0.70 | 0.662 | 0.50 |
| gbdt | random 5-fold | 214 | 0.888 | 0.27 | 0.12 | 0.940 | 0.91 |
| gbdt | leave-one-family-out | 214 | 0.590 | 0.51 | 0.41 | 0.758 | 0.72 |

GBDT top features (capture-time set): proto_bytes 0.27, n_instructions 0.20, n_entry_instructions 0.17, total_tensor_bytes 0.06, n_params 0.06, n_broadcast 0.03, dot_flops 0.03, elementwise_bytes 0.02, frac_f32 0.02, param_bytes 0.02
- Spearman(log wall, n_instructions) = 0.839
- Spearman(log wall, proto_bytes) = 0.827
- Spearman(log wall, total_tensor_bytes) = 0.724
- Spearman(log wall, dot_flops) = 0.617
- Spearman(log wall, largest_tensor_bytes) = 0.544
- modules sharing the exact same instruction count (groups of ≥3: 22): compile-time max/min within a group median 2.89×, max 79.7×

## 6. Execution latency: neuron-profile vs framework ExecuteTime

profiled NEFFs: 0 rows (by arm: {})
no paired latency rows; profile keys: 
