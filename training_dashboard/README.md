# Training Dashboard

Static frontend for watching autoresearch training logs.

Run it from the repository root so the page can fetch `remote_training_runs`:

```bash
python -m http.server 8787
```

Open:

```text
http://127.0.0.1:8787/training_dashboard/
```

The dashboard parses:

- `step ... train_loss ...`
- `val_loss: ... step: ...`
- `observable: step=... key=value ...`
- final metric lines such as `val_bpb: ...`

Use live mode to poll a log URL while training is running, or load a saved `.log`
file with the file picker.
