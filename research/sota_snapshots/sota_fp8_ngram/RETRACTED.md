# RETRACTED — fp8-MLP was a GPU-contention artifact, not a real win.
Interleaved-GPU paired A/B (round 36) showed fp8 vs bf16 Δ=+0.0002 (neutral), MFU equal.
The earlier "win" came from putting bf16 controls on slower/contended shared-cluster GPUs.
TRUE SOTA = the bf16 n-gram config in ../sota_ngram_2p20/ (val_bpb ~0.9546, best 0.9521).
