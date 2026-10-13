# Phase 2 report

Generated: 2026-10-10T11:43:15

Output index: `project/data/faces_index.parquet` (87705 rows)

## Extraction per dataset

| dataset | attempted | saved | skipped (no face) | haar fallback used | errors |
|---|---:|---:|---:|---:|---:|
| crema_d | 37210 | 37205 | 5 | 0 | 1 |
| fer2013 | 35887 | 35887 | 0 | 0 | 0 |
| jaffe | 213 | 213 | 0 | 0 | 0 |
| ravdess | 14400 | 14400 | 0 | 0 | 0 |

fer2013 is passthrough (no detection): 'attempted' counts images, 'saved' counts index rows.

## Counts per dataset and class (rows in faces_index)

```
label    angry  disgust  fear  happy  neutral   sad  surprise
dataset                                                      
crema_d   6355     6355  6355   6355     5435  6350         0
fer2013   4953      547  5121   8989     6198  6077      4002
jaffe       30       29    32     31       30    31        30
ravdess   1920     1920  1920   1920     2880  1920      1920
```

## Contact sheets

- `project/reports/contact_crema_d.png`
- `project/reports/contact_fer2013.png`
- `project/reports/contact_jaffe.png`
- `project/reports/contact_ravdess.png`

Elapsed: 29.2 min
