# Golden TUI frames

These files are ANSI snapshots of the Textual compositor at the layout sizes
that matter for Kinesis:

- `chat-normal-120x40.ansi` — wide layout with the sidebar
- `chat-normal-80x24.ansi` — sidebar auto-hidden, one pane
- `chat-normal-60x24.ansi` — narrow one-pane layout
- `chat-confirm-80x24.ansi` — confirmation banner above the input
- `too-small-60x18.ansi` — minimum-size overlay

They are asserted by `tests/test_tui.py` at exact character/style level, so
regressions in chrome height, tab clipping, banner wrapping, footer collisions,
and the too-small state fail loudly.

Regenerate after an intentional visual change:

```powershell
$env:UPDATE_SNAPSHOTS = "1"
.venv\Scripts\python.exe -m pytest tests\test_tui.py -q -k golden
Remove-Item Env:\UPDATE_SNAPSHOTS
```

Review the resulting diff before committing it.
