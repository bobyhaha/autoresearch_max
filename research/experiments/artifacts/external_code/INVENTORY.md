# Immutable External-Code Inventory

Captured and verified on `2026-07-29`. These files are external mechanism or
evaluation references only. None is a governed local OPHIS SOTA.

| Reference | Commit | Artifact | Bytes | SHA-256 | Scope |
|---|---|---:|---:|---|---|
| Recursive nanochat autoresearch | `a962ec43e2e3d7c018e59a2ece623fe6e232fdfb` | `pap_recursive_nanochat_autoresearch/recursive-a962ec43.tar.zst` | 226,273 | `4465847c4b9c12a2d51bbbe9349c4a17444255347b8c9a1de245920729281d48` | Published autoresearch reference |
| Recursive nanochat autoresearch | `a962ec43e2e3d7c018e59a2ece623fe6e232fdfb` | `pap_recursive_nanochat_autoresearch/recursive-a962ec43.bundle` | 532,933 | `432438abbcbcc22fabacf60294aefb2257e1c356d61536c7ad9b81224834b0d8` | Complete Git-history bundle |
| Gated Attention | `f4c2a5f6ffd6ec709e0c60072c95ed4f5ce5b5d2` | `pap_r2_arch_gated_attention/gated-attention-f4c2a5f6.tar.zst` | 532,998 | `ac9e5d5d4270ca88071500343b525be2116dbfd68c0c422411d35f2a078270fb` | Ranked elementwise/headwise gate source |
| FlashAttention | `849f660f73b176e5ad5670e7f822c7fa9f3eaf8b` | `pap_flashattention4/flash-attention-849f660f.tar.zst` | 13,287,461 | `deb295841d111a2140f8e469db3997e62aa99fce08d91aa7212d7d41acc8c5db` | Canonical FA3/FA4 top-level source; three gitlinks recorded, not vendored |
| FlashAttention-3 varlen API | `c75d019dea9d910312974417bc28f190dfdda6d9` | `pap_flash_attention_3_neurips2024/flash-attention-c75d019d.tar.zst` | 13,287,191 | `89a16aa92cb9b84bcf6c8abab576e3304216092f15ee3e0340f734ff1970d9e8` | Exact official Hopper varlen cumulative-offset API and tests; three gitlinks recorded, not vendored |
| FlashAttention-3 varlen API | `c75d019dea9d910312974417bc28f190dfdda6d9` | `pap_flash_attention_3_neurips2024/flash-attention-c75d019d.bundle` | 20,308,909 | `d5d096b964a66625b178a19e044c36a4fdbdcbe83720ac5f5976ee3221dc5e99` | Complete Git history through the exact API capsule commit |
| FlashAttention-3 | `NeurIPS 2024` | `pap_flash_attention_3_neurips2024/FlashAttention-3_NeurIPS_2024.pdf` | 697,391 | `73e785a3dc6e378d8940dbc2f682f06e1fa9b0ed26295aeb0a6e7ce6bb2f8291` | Official proceedings paper; Hopper algorithm context, not local sidecar efficacy |
| KernelBench-Verified | `3fdf6fec7372a4d0cb682635f00e7bdcbc55d50e` | `pap_kernelbench_verified/kernel-bench-verified-3fdf6fec.tar.zst` | 1,175,772 | `9d3439ba947e677acd0ea12dd37f35588504f7c7c971e139b1cb5c74c403a013` | Preferred adversarial kernel-evaluation control |
| Cut Cross-Entropy | `b7a02791b234e187b524fb1dba6a812d521b203a` | `pap_cut_cross_entropy/ml-cross-entropy-b7a02791.tar.zst` | 412,865 | `4222a83d4d25304beac914424bb43d108c9894f78c4d6fb44f9fbf776f28ec94` | External CCE systems reference |
| Original KernelBench | `423217d9fda91e0c2d67e4a43bf62f96f6d104f1` | `pap_kernelbench_original/kernelbench-423217d9.tar.zst` | 1,578,780 | `a7fa556288a7cce83c04986bb0e65faecae3551c4ba39ec7da9cfdf19f8ad041` | Historical comparison only; explicitly superseded-risk |
| DeepSeek Engram | `fb7f84a21f91223715394a33a1dc24bbfb7f788e` | `pap_deepseek_engram_2026/engram-fb7f84a2.tar.zst` | 1,784,714 | `789b81408ab98b4cb90a2b00f4f6a15ded1651f13972cd4498edcd5b52ac1061` | Canonicalized n-gram key and conditional-memory source |
| DeepSeek Engram | `fb7f84a21f91223715394a33a1dc24bbfb7f788e` | `pap_deepseek_engram_2026/engram-fb7f84a2.bundle` | 2,352,059 | `03114f299985aba9267b79299a2331d96194c0d50eb89a5f6e6b5eec3d40c0f0` | Complete Git-history bundle |
| DeepSeek Engram | `fb7f84a21f91223715394a33a1dc24bbfb7f788e` | `pap_deepseek_engram_2026/Engram_paper.pdf` | 770,776 | `1e8ae53dc527d4db264b40a7996780b0044914443c0ce5fd54f898c405cab94f` | Upstream-commit primary paper |
| GPAS | `31980688f4cbb1b0cff59bca9077e6fc52dab3f0` | `pap_gpas_neurips2025/gpas-31980688.tar.zst` | 92,641 | `1177e6766f53cf0bc59b880a96ca09d7732984db62f8d54b9bf5083b1b45acf0` | NeurIPS 2025 gradient-preserving residual activation-scaling source |
| GPAS | `31980688f4cbb1b0cff59bca9077e6fc52dab3f0` | `pap_gpas_neurips2025/gpas-31980688.bundle` | 98,272 | `8e2abb5ba02b16639d92cb532b3134c5fa4bcce0f43156585fa2e591294edaf1` | Complete Git-history bundle |
| GPAS | `NeurIPS 2025` | `pap_gpas_neurips2025/GPAS_NeurIPS_2025.pdf` | 1,407,649 | `d644ed31e31a693fa889e01655e94cfea1cdb0c312901ee11b001af68b14a35e` | Official proceedings paper |

Total immutable source payload: **58,546,684 bytes**.

Each new `manifest.json` records upstream/default-branch resolution, exact
commit and tree objects, capture time, deterministic archive recipe, license
path/hash, selected mechanism or evaluation file hashes, integrity checks,
submodule boundaries when applicable, and scope/adoption caveats.
