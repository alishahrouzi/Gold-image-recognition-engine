# Dataset 1 DataLoader Benchmark (S1.12)

This report measures the **data pipeline only** (load → optional train augmentation → preprocess → collate → optional GPU copy). It does not run an Encoder or a training loop.

## 1. Hardware

| Field | Value |
| --- | --- |
| Python | 3.12.10 |
| PyTorch | 2.6.0+cu124 |
| Platform | Windows-11-10.0.26200-SP0 |
| CUDA available | true |
| CUDA version | 12.4 |
| GPU | NVIDIA GeForce GTX 1650 |
| VRAM total (MiB) | 4095.6875 |
| System RAM total (MiB) | 32189.6719 |
| Process RSS at start (MiB) | 459.3047 |
| GPU utilization measurable | false |

GPU utilization is not reported because short CPU→GPU copies do not produce a reliable utilization sample from this process. torch.cuda.utilization() requires CUPTI and is often unavailable; instant NVML/nvidia-smi snapshots during a few dozen transfers are noisy and would be misleading. gpu_utilization_percent is therefore null.

Process RAM below is RSS for this Python process, not OS-wide memory.

## 2. Dataset

| Field | Value |
| --- | --- |
| Name | dataset1 |
| Dataset path | E:\Programming Project\Visual Recognition Engine\dataset\ai-tool-pool-jewelry-vision |
| Manifest | E:\Programming Project\Visual Recognition Engine\Gold-image-recognition-engine\reports\dataset\dataset1_manifest.csv |
| Splits | train, valid, test |
| Split sizes | {"train": 4328, "valid": 429, "test": 212} |

The existing Dataset 1 manifest is the source of truth. Source images and the manifest are not modified.

## 3. Benchmark methodology

- High-resolution monotonic clock (`time.perf_counter`).
- Requested warmup batches: 5.
- Requested measurement batches: 30.
- Warmup latencies are discarded (first-batch latency is still recorded).
- Measurement count is capped per configuration if a split is too small.
- Train uses the existing S1.9 `TrainingAugmentor` defaults.
- Valid and test use `ImagePreprocessor` only (no random augmentation).
- `dataloader` stage never copies tensors to GPU.
- `dataloader_gpu` stage copies `batch['image']` with `tensor.to(device, non_blocking=pin_memory)` after CUDA synchronize.
- No Encoder, loss, optimizer, or backward pass.
- Seed: 2026.
- Device: cuda:0.

Preprocessing contract:

- RGB → resize 224×224 → ImageNet normalize → float32 [3, 224, 224]
- interpolation: `bilinear`
- mean: `[0.485, 0.456, 0.406]`
- std: `[0.229, 0.224, 0.225]`

Train augmentation: enabled via S1.9 defaults (seed=2026). Valid/test: deterministic, augmentation disabled.

## 4. Configuration matrix

| split | stage | batch_size | num_workers | pin_memory | persistent_workers | status |
| --- | --- | --- | --- | --- | --- | --- |
| train | dataloader | 8 | 0 | false | false | PASS |
| train | dataloader_gpu | 8 | 0 | false | false | PASS |
| train | dataloader | 8 | 2 | false | false | PASS |
| train | dataloader_gpu | 8 | 2 | false | false | PASS |
| train | dataloader | 8 | 4 | false | false | PASS |
| train | dataloader_gpu | 8 | 4 | false | false | PASS |
| train | dataloader | 16 | 0 | false | false | PASS |
| train | dataloader_gpu | 16 | 0 | false | false | PASS |
| train | dataloader | 16 | 2 | false | false | PASS |
| train | dataloader_gpu | 16 | 2 | false | false | PASS |
| train | dataloader | 16 | 4 | false | false | PASS |
| train | dataloader_gpu | 16 | 4 | false | false | PASS |
| train | dataloader | 32 | 0 | false | false | PASS |
| train | dataloader_gpu | 32 | 0 | false | false | PASS |
| train | dataloader | 32 | 2 | false | false | PASS |
| train | dataloader_gpu | 32 | 2 | false | false | PASS |
| train | dataloader | 32 | 4 | false | false | PASS |
| train | dataloader_gpu | 32 | 4 | false | false | PASS |
| train | dataloader | 64 | 0 | false | false | PASS |
| train | dataloader_gpu | 64 | 0 | false | false | PASS |
| train | dataloader | 64 | 2 | false | false | PASS |
| train | dataloader_gpu | 64 | 2 | false | false | PASS |
| train | dataloader | 64 | 4 | false | false | PASS |
| train | dataloader_gpu | 64 | 4 | false | false | PASS |
| train | dataloader | 16 | 0 | true | false | PASS |
| train | dataloader_gpu | 16 | 0 | true | false | PASS |
| train | dataloader | 16 | 2 | true | false | PASS |
| train | dataloader_gpu | 16 | 2 | true | false | PASS |
| train | dataloader | 32 | 0 | true | false | PASS |
| train | dataloader_gpu | 32 | 0 | true | false | PASS |
| train | dataloader | 32 | 2 | true | false | PASS |
| train | dataloader_gpu | 32 | 2 | true | false | PASS |
| train | dataloader | 16 | 2 | true | true | PASS |
| train | dataloader_gpu | 16 | 2 | true | true | PASS |
| valid | dataloader | 16 | 0 | false | false | PASS |
| valid | dataloader_gpu | 16 | 0 | false | false | PASS |
| test | dataloader | 16 | 0 | false | false | PASS |
| test | dataloader_gpu | 16 | 0 | false | false | PASS |

## 5. CPU / RAM results

| split | stage | batch | workers | pin | status | RAM before (MiB) | RAM after warmup (MiB) | RAM peak (MiB) | RAM delta (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | dataloader | 8 | 0 | false | PASS | 471.9062 | 487.0547 | 490.1719 | 18.2656 |
| train | dataloader_gpu | 8 | 0 | false | PASS | 486.9961 | 576.6523 | 576.6797 | 89.6836 |
| train | dataloader | 8 | 2 | false | PASS | 576.6914 | 577.0859 | 577.0859 | 0.3945 |
| train | dataloader_gpu | 8 | 2 | false | PASS | 577.0625 | 581.8203 | 581.8203 | 4.7578 |
| train | dataloader | 8 | 4 | false | PASS | 577.1992 | 577.2812 | 577.3086 | 0.1094 |
| train | dataloader_gpu | 8 | 4 | false | PASS | 577.2109 | 581.8945 | 581.9023 | 4.6914 |
| train | dataloader | 16 | 0 | false | PASS | 577.2109 | 587.4492 | 590.3047 | 13.0938 |
| train | dataloader_gpu | 16 | 0 | false | PASS | 590.3047 | 590.3047 | 590.3047 | 0.0000 |
| train | dataloader | 16 | 2 | false | PASS | 590.3047 | 590.3594 | 590.3633 | 0.0586 |
| train | dataloader_gpu | 16 | 2 | false | PASS | 590.3438 | 599.5469 | 599.5508 | 9.2070 |
| train | dataloader | 16 | 4 | false | PASS | 590.3125 | 590.3867 | 590.4297 | 0.1172 |
| train | dataloader_gpu | 16 | 4 | false | PASS | 590.3242 | 599.6055 | 599.6172 | 9.2930 |
| train | dataloader | 32 | 0 | false | PASS | 590.3242 | 611.2305 | 611.2617 | 20.9375 |
| train | dataloader_gpu | 32 | 0 | false | PASS | 611.2500 | 611.2500 | 611.2578 | 0.0078 |
| train | dataloader | 32 | 2 | false | PASS | 611.2578 | 611.3164 | 611.3203 | 0.0625 |
| train | dataloader_gpu | 32 | 2 | false | PASS | 611.3008 | 629.6875 | 629.6953 | 18.3945 |
| train | dataloader | 32 | 4 | false | PASS | 611.2695 | 611.3828 | 611.3906 | 0.1211 |
| train | dataloader_gpu | 32 | 4 | false | PASS | 611.2852 | 629.7617 | 629.7852 | 18.5000 |
| train | dataloader | 64 | 0 | false | PASS | 611.3047 | 649.4062 | 649.4102 | 38.1055 |
| train | dataloader_gpu | 64 | 0 | false | PASS | 649.4102 | 649.4102 | 649.4102 | 0.0000 |
| train | dataloader | 64 | 2 | false | PASS | 649.3945 | 649.4766 | 649.4766 | 0.0820 |
| train | dataloader_gpu | 64 | 2 | false | PASS | 649.4414 | 686.2109 | 686.2109 | 36.7695 |
| train | dataloader | 64 | 4 | false | PASS | 649.4414 | 649.5117 | 649.5391 | 0.0977 |
| train | dataloader_gpu | 64 | 4 | false | PASS | 649.4023 | 686.2578 | 686.2578 | 36.8555 |
| train | dataloader | 16 | 0 | true | PASS | 649.4766 | 618.0039 | 649.4766 | 0.0000 |
| train | dataloader_gpu | 16 | 0 | true | PASS | 618.0039 | 618.0469 | 618.0469 | 0.0430 |
| train | dataloader | 16 | 2 | true | PASS | 618.0469 | 634.2031 | 650.2031 | 32.1562 |
| train | dataloader_gpu | 16 | 2 | true | PASS | 650.1758 | 650.2383 | 654.4297 | 4.2539 |
| train | dataloader | 32 | 0 | true | PASS | 650.1758 | 700.6250 | 700.6250 | 50.4492 |
| train | dataloader_gpu | 32 | 0 | true | PASS | 700.6250 | 700.6250 | 700.6289 | 0.0039 |
| train | dataloader | 32 | 2 | true | PASS | 700.6289 | 732.6953 | 764.6953 | 64.0664 |
| train | dataloader_gpu | 32 | 2 | true | PASS | 764.6328 | 764.6953 | 764.6953 | 0.0625 |
| train | dataloader | 16 | 2 | true | PASS | 764.6328 | 764.6953 | 764.6953 | 0.0625 |
| train | dataloader_gpu | 16 | 2 | true | PASS | 764.6484 | 764.6953 | 764.6953 | 0.0469 |
| valid | dataloader | 16 | 0 | false | PASS | 764.6328 | 743.6953 | 764.6328 | 0.0000 |
| valid | dataloader_gpu | 16 | 0 | false | PASS | 742.4453 | 743.6953 | 747.4453 | 5.0000 |
| test | dataloader | 16 | 0 | false | PASS | 747.4453 | 743.6953 | 747.4453 | 0.0000 |
| test | dataloader_gpu | 16 | 0 | false | PASS | 743.6953 | 743.6953 | 747.4453 | 3.7500 |

## 6. GPU / VRAM results

| split | stage | batch | workers | pin | status | alloc (MiB) | reserved (MiB) | peak alloc (MiB) | peak reserved (MiB) | GPU util % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | dataloader_gpu | 8 | 0 | false | PASS | 4.5938 | 20.0000 | 9.1875 | 20.0000 | null |
| train | dataloader_gpu | 8 | 2 | false | PASS | 4.5938 | 20.0000 | 9.1875 | 20.0000 | null |
| train | dataloader_gpu | 8 | 4 | false | PASS | 4.5938 | 20.0000 | 9.1875 | 20.0000 | null |
| train | dataloader_gpu | 16 | 0 | false | PASS | 9.1875 | 20.0000 | 18.3750 | 20.0000 | null |
| train | dataloader_gpu | 16 | 2 | false | PASS | 9.1875 | 20.0000 | 18.3750 | 20.0000 | null |
| train | dataloader_gpu | 16 | 4 | false | PASS | 9.1875 | 20.0000 | 18.3750 | 20.0000 | null |
| train | dataloader_gpu | 32 | 0 | false | PASS | 18.3750 | 40.0000 | 36.7500 | 40.0000 | null |
| train | dataloader_gpu | 32 | 2 | false | PASS | 18.3750 | 40.0000 | 36.7500 | 40.0000 | null |
| train | dataloader_gpu | 32 | 4 | false | PASS | 18.3750 | 40.0000 | 36.7500 | 40.0000 | null |
| train | dataloader_gpu | 64 | 0 | false | PASS | 36.7500 | 76.0000 | 73.5000 | 76.0000 | null |
| train | dataloader_gpu | 64 | 2 | false | PASS | 36.7500 | 76.0000 | 73.5000 | 76.0000 | null |
| train | dataloader_gpu | 64 | 4 | false | PASS | 36.7500 | 76.0000 | 73.5000 | 76.0000 | null |
| train | dataloader_gpu | 16 | 0 | true | PASS | 9.1875 | 20.0000 | 18.3750 | 20.0000 | null |
| train | dataloader_gpu | 16 | 2 | true | PASS | 9.1875 | 20.0000 | 18.3750 | 20.0000 | null |
| train | dataloader_gpu | 32 | 0 | true | PASS | 18.3750 | 40.0000 | 36.7500 | 40.0000 | null |
| train | dataloader_gpu | 32 | 2 | true | PASS | 18.3750 | 40.0000 | 36.7500 | 40.0000 | null |
| train | dataloader_gpu | 16 | 2 | true | PASS | 9.1875 | 20.0000 | 18.3750 | 20.0000 | null |
| valid | dataloader_gpu | 16 | 0 | false | PASS | 7.4648 | 20.0000 | 18.3750 | 20.0000 | null |
| test | dataloader_gpu | 16 | 0 | false | PASS | 2.2969 | 20.0000 | 18.3750 | 20.0000 | null |

## 7. Throughput comparison

| split | stage | batch | workers | pin | persistent | first ms | mean ms | median ms | p95 ms | batches/s | images/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | dataloader | 8 | 0 | false | false | 3304.2024 | 3067.5438 | 3126.2175 | 3308.8925 | 0.3260 | 2.6079 |
| train | dataloader_gpu | 8 | 0 | false | false | 3430.0448 | 2967.0739 | 3034.5821 | 3254.0243 | 0.3370 | 2.6963 |
| train | dataloader | 8 | 2 | false | false | 3505.7004 | 1416.4709 | 1500.4582 | 1829.1229 | 0.7060 | 5.6478 |
| train | dataloader_gpu | 8 | 2 | false | false | 3555.8508 | 1451.5921 | 1423.4126 | 1916.7121 | 0.6889 | 5.5112 |
| train | dataloader | 8 | 4 | false | false | 4679.1052 | 984.3487 | 435.7549 | 2803.5197 | 1.0159 | 8.1272 |
| train | dataloader_gpu | 8 | 4 | false | false | 4748.0088 | 992.0250 | 436.7905 | 2788.9146 | 1.0080 | 8.0643 |
| train | dataloader | 16 | 0 | false | false | 7551.1511 | 2824.9911 | 501.9410 | 6395.4464 | 0.3540 | 5.6637 |
| train | dataloader_gpu | 16 | 0 | false | false | 7812.3470 | 2592.7903 | 369.1785 | 5918.1833 | 0.3857 | 6.1710 |
| train | dataloader | 16 | 2 | false | false | 7166.4023 | 1155.5830 | 64.1354 | 5456.0292 | 0.8654 | 13.8458 |
| train | dataloader_gpu | 16 | 2 | false | false | 6992.1346 | 1151.6400 | 63.3081 | 5435.7180 | 0.8683 | 13.8932 |
| train | dataloader | 16 | 4 | false | false | 9221.8104 | 682.0022 | 0.2293 | 6423.4635 | 1.4663 | 23.4603 |
| train | dataloader_gpu | 16 | 4 | false | false | 8651.2098 | 678.5529 | 7.9679 | 6344.8731 | 1.4737 | 23.5796 |
| train | dataloader | 32 | 0 | false | false | 14133.7111 | 3232.0195 | 598.5452 | 11860.2101 | 0.3094 | 9.9009 |
| train | dataloader_gpu | 32 | 0 | false | false | 13820.8991 | 3080.9711 | 327.0514 | 11812.1926 | 0.3246 | 10.3863 |
| train | dataloader | 32 | 2 | false | false | 13939.5535 | 1476.3255 | 271.2874 | 10729.5234 | 0.6774 | 21.6754 |
| train | dataloader_gpu | 32 | 2 | false | false | 13883.8880 | 1470.6921 | 256.5830 | 10462.5285 | 0.6800 | 21.7585 |
| train | dataloader | 32 | 4 | false | false | 17416.4533 | 930.2944 | 0.2822 | 6341.2107 | 1.0749 | 34.3977 |
| train | dataloader_gpu | 32 | 4 | false | false | 17098.0871 | 941.1321 | 14.0881 | 6476.6437 | 1.0626 | 34.0016 |
| train | dataloader | 64 | 0 | false | false | 26495.0021 | 11463.7737 | 1759.0149 | 29099.8109 | 0.0872 | 5.5828 |
| train | dataloader_gpu | 64 | 0 | false | false | 27202.3554 | 11183.0305 | 1740.7655 | 28837.1678 | 0.0894 | 5.7230 |
| train | dataloader | 64 | 2 | false | false | 26684.8523 | 5837.9403 | 611.7983 | 21255.4905 | 0.1713 | 10.9628 |
| train | dataloader_gpu | 64 | 2 | false | false | 26275.7408 | 5848.3457 | 574.6019 | 21428.4886 | 0.1710 | 10.9433 |
| train | dataloader | 64 | 4 | false | false | 32617.4827 | 4097.1180 | 0.3316 | 23997.1130 | 0.2441 | 15.6207 |
| train | dataloader_gpu | 64 | 4 | false | false | 33088.9990 | 4103.8314 | 28.6546 | 24517.7582 | 0.2437 | 15.5952 |
| train | dataloader | 16 | 0 | true | false | 7332.6267 | 2554.6875 | 364.2101 | 5804.3120 | 0.3914 | 6.2630 |
| train | dataloader_gpu | 16 | 0 | true | false | 7121.1642 | 2589.9868 | 371.9334 | 6000.9841 | 0.3861 | 6.1776 |
| train | dataloader | 16 | 2 | true | false | 7156.6386 | 1158.5029 | 65.6746 | 5510.3774 | 0.8632 | 13.8109 |
| train | dataloader_gpu | 16 | 2 | true | false | 7156.3861 | 1162.6763 | 62.3683 | 5484.4834 | 0.8601 | 13.7614 |
| train | dataloader | 32 | 0 | true | false | 13511.1740 | 3055.0768 | 323.5992 | 11842.8278 | 0.3273 | 10.4744 |
| train | dataloader_gpu | 32 | 0 | true | false | 13701.2296 | 3051.9573 | 322.7200 | 11892.3276 | 0.3277 | 10.4851 |
| train | dataloader | 32 | 2 | true | false | 13288.9258 | 1467.8781 | 266.3715 | 10703.8320 | 0.6813 | 21.8002 |
| train | dataloader_gpu | 32 | 2 | true | false | 13355.3211 | 1487.4346 | 269.3487 | 10850.1605 | 0.6723 | 21.5136 |
| train | dataloader | 16 | 2 | true | true | 6886.7408 | 1133.5489 | 65.4046 | 5342.5024 | 0.8822 | 14.1150 |
| train | dataloader_gpu | 16 | 2 | true | true | 6958.9745 | 1145.7231 | 65.3271 | 5472.9966 | 0.8728 | 13.9650 |
| valid | dataloader | 16 | 0 | false | false | 640.2488 | 596.4663 | 707.9656 | 957.8974 | 1.6765 | 26.5960 |
| valid | dataloader_gpu | 16 | 0 | false | false | 663.9260 | 452.6040 | 559.2243 | 832.9163 | 2.2094 | 35.0497 |
| test | dataloader | 16 | 0 | false | false | 519.8447 | 687.8186 | 815.8028 | 1075.0829 | 1.4539 | 21.3235 |
| test | dataloader_gpu | 16 | 0 | false | false | 332.4277 | 485.3468 | 607.6644 | 803.3092 | 2.0604 | 30.2189 |

## 8. OOM configurations

None. No CUDA OOM was recorded for the DataLoader / transfer stages.

Failed (non-OOM) configurations: none

## 9. Recommended DataLoader configuration

| Field | Value |
| --- | --- |
| Largest successful DataLoader batch | 64 |
| Recommended DataLoader batch size | 8 |
| Selection pool | train_dataloader_gpu |
| Recommended split | train |
| Recommended stage | dataloader_gpu |
| Recommended num_workers | 4 |
| Recommended pin_memory | false |
| Recommended persistent_workers | false |
| Recommended images/s | 8.0643 |

This is a DataLoader-safe batch size, not the final model-training batch size. The largest successful DataLoader batch only means the CPU/GPU copy of preprocessed tensors fit; an Encoder, loss, and optimizer will consume additional VRAM and must be re-benchmarked before choosing a training batch size.

## 10. Limitations

- Throughput is for the data pipeline, not training step time.
- GPU utilization is not claimed (see hardware note).
- Worker processes on Windows use spawn; first-batch latency includes worker start.
- Valid/test use a smaller matrix than train.
- A 4 GB GPU will be dominated by the Encoder later; tensor-only VRAM is not representative.
- Persistent workers are sampled, not fully crossed with every batch size.

## 11. Important note about final training batch size

**This is a DataLoader-safe batch size, not the final model-training batch size.**

After the Encoder and optimizer are introduced, re-run a training-step VRAM benchmark. Do not treat the largest successful DataLoader batch as the training batch size.

