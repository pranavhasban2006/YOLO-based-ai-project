# Audio phase 1 report (2026-10-12 15:47)

Index: `project/data/audio_index.csv`, 43821 clips.

## Clips per dataset and label

| dataset | speakers | angry | disgust | fear | happy | neutral | sad | surprise | total |
|---|---|---|---|---|---|---|---|---|---|
| ravdess | 24 | 192 | 192 | 192 | 192 | 288 | 192 | 192 | 1440 |
| crema_d | 91 | 1271 | 1271 | 1271 | 1271 | 1087 | 1270 | 0 | 7441 |
| tess | 3 | 400 | 400 | 400 | 400 | 400 | 400 | 400 | 2800 |
| savee | 4 | 60 | 60 | 60 | 60 | 120 | 60 | 60 | 480 |
| meld | 304 | 1607 | 361 | 358 | 2308 | 6434 | 1002 | 1636 | 13706 |
| esd | 10 | 3500 | 0 | 0 | 3500 | 3500 | 3500 | 3500 | 17500 |
| emodb | 10 | 127 | 46 | 69 | 71 | 79 | 62 | 0 | 454 |
| **all** | 446 | 7157 | 2330 | 2350 | 7802 | 11908 | 6486 | 5788 | 43821 |

## Classes absent per dataset

- crema_d: no clips for surprise
- esd: no clips for disgust, fear
- emodb: no clips for surprise

Known by design: CREMA-D has no surprise; EMO-DB has no surprise (and boredom is excluded, see DECISIONS.md).

## Excluded clips (not in the index)

Total excluded: 82
- boredom_excluded: 81
- silent peak 0.00e+00: 1
