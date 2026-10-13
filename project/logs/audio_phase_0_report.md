# Audio phase 0 report (2026-10-12)

Audit: `code/audit_audio_datasets.py` (read-only walk of `datasets/`), raw output in `logs/audio_phase0_audit.json`.
Nothing under `datasets/` was downloaded, extracted, moved, or deleted by this phase.

| dataset | status | folder | files | audio files used | size GB | zero-byte | partial | unextracted zips | reason |
|---|---|---|---|---|---|---|---|---|---|
| ravdess | **COMPLETE** | RAVDESS | 4349 | 1440 speech wav (Audio_Speech) + 2880 speech mp4 (Video_Speech) | 27.29 | 0 | 0 | 3 song zips (not used) | speech subset 1440/1440 wav; song zips skipped by earlier decision |
| crema_d | **COMPLETE** | crema_d | 44813 | 7442 wav (AudioWAV) | 6.93 | 131 (all in `.git`) | 0 | 0 | 7442/7442 wav; zero-byte items are git-lfs temp files and `.git/index.lock`, not data |
| tess | **COMPLETE** | tess | 2801 | 2800 wav | 0.26 | 0 | 0 | 0 | 2800/2800 wav (+ `OA_bite_neutral.wav`) |
| savee | **COMPLETE** | savee | 482 | 480 wav | 0.11 | 0 | 0 | 0 | 480/480 wav (4 speakers x 120); `labels.csv` present; 16 kHz mirror |
| meld | **COMPLETE** | meld | 15928 | 15908 mp4 (MELD.Raw: 9989 train + 1112 dev + 4807 test folder) | 30.42 | 2 (logs only) | 0 | 0 (tarball `MELD.Raw.tar.gz` kept, not a zip) | 15908 mp4 present; 3 CSVs present; tarball size matches 10,878,146,150 bytes |
| esd | **COMPLETE** | esd | 17525 | 17500 wav (English speakers 0011-0020, 1750 each) | 3.72 | 1 (`gdown.log`) | 0 | 1 (`ESD.zip`, extracted content already present) | 17500/17500 expected for 10 English speakers; Mandarin 0001-0010 not extracted (not used) |
| emodb (extra, not in the list) | **COMPLETE** | emodb | 2105 | 535 wav | 0.08 | 1 (`silb/13a01Wb.silb`, not audio) | 0 | 1 (`raw/download.zip`, extracted content present) | 535/535 wav; boredom excluded later per DECISIONS |

## Notes

- **Duplicate-folder check.** Windows folder names are case-insensitive, so the audit reads only the first matching folder per dataset. An earlier run double-counted RAVDESS; that run was discarded.
- **Ambiguous MELD counts.** The folder counts (9989 / 1112 / 4807 mp4) do not match the CSV row counts (9989 / 1109 / 2610). The prior download report says this was not investigated; the phase 1 index will use the CSVs for labels.
- **Stale runner.** The headless `claude -p` run started by `run_audio_phases.ps1` (PID 48636, 08:29) produced no output after 08:31 and wrote no state or report files. It was not killed by this phase.

## Datasets available for phase 1

COMPLETE and used: ravdess, crema_d, tess, savee, meld, esd. EMO-DB is also COMPLETE and is recorded, but it was not in the phase 0 list.
