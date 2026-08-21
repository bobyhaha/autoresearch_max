# GPU Occupation Check

Use this skill before launching a GPU training run, especially remote H200 runs, when the user wants to avoid colliding with an already occupied GPU.

## Goals

1. Check current GPU occupation.
2. If any GPU is occupied, show the occupation information and let the user decide what to do and which GPU to run on.
3. Skip the user conversation during an already continuous training process.

## Procedure

### 1. Determine Whether This Is Interactive

Treat the run as interactive unless there is clear evidence that training is already in a continuous process.

Continuous training examples:

- The current task is resuming, monitoring, or continuing an existing training run.
- A non-interactive orchestration script is already running.
- The user explicitly asked to continue training without interruption.
- The environment or script already provides a selected GPU through `CUDA_VISIBLE_DEVICES`, `GPU_ID`, or equivalent configuration.

If this is continuous training, do not stop for a user decision. Report the GPU state in logs or the final summary, then continue with the configured GPU selection.

### 2. Check GPU Occupation

For local machines, run:

```bash
nvidia-smi
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
```

For H200 remote runs, prefer the repository's configured SSH helper if available:

```bash
./run_h200_training.sh
```

or run the equivalent check through the H200 helper:

```bash
"$HELPER" run "nvidia-smi && nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv"
```

If `nvidia-smi` is unavailable or errors, report that clearly and do not infer GPU availability.

### 3. Summarize Occupation

When a GPU is occupied, present concise information:

- GPU index, name, and total/used memory.
- Process PID.
- Process name or command.
- Used GPU memory.
- Owner/user if available.

If possible, include a mapping from GPU UUID to GPU index by combining:

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
```

with:

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
```

### 4. Ask Before Proceeding

If this is an interactive run and any GPU is occupied, stop before launching training and ask the user what to do.

Ask for:

- Whether to wait, abort, or proceed anyway.
- Which GPU to run on.

Do not kill GPU processes unless the user explicitly requests it. Do not assume an occupied process is safe to terminate.

Suggested prompt:

```text
GPU occupation detected:
<summary>

Which GPU should I run on, and should I wait, abort, or proceed anyway?
```

### 5. Continuous Training Behavior

During continuous training, skip the conversation. Do not ask follow-up questions merely because another GPU process exists.

Instead:

- Log the GPU occupation.
- Keep the existing configured GPU selection.
- Continue the training orchestration.
- If no GPU selection is configured and every GPU appears occupied, fail fast with a clear error rather than killing processes.

### 6. Safety Rules

- Never kill GPU processes as part of this skill without explicit user approval.
- Prefer selecting an available GPU over changing or terminating existing jobs.
- If the occupation data is ambiguous, report uncertainty instead of guessing.
- Keep output concise enough that the user can decide quickly.
