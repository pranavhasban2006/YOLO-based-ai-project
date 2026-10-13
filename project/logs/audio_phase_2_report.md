# Audio phase 2 report (2026-10-12 16:01)

Clips written to `project/data/audio_16k/`: 23174

Conversion: skipped_existing 720, ok 22454

## Size cap

- crema_d: capped 7441 -> 6000 clips (stratified by label, rare classes first, seed 42): neutral 1000/1087, sad 1000/1270, angry 1000/1271, disgust 1000/1271, fear 1000/1271, happy 1000/1271
- esd: capped 17500 -> 6000 clips (stratified by label, rare classes first, seed 42): angry 1200/3500, happy 1200/3500, neutral 1200/3500, sad 1200/3500, surprise 1200/3500
- meld: capped 13706 -> 6000 clips (stratified by label, rare classes first, seed 42): fear 358/358, disgust 361/361, sad 1002/1002, angry 1069/1607, surprise 1070/1636, happy 1070/2308, neutral 1070/6434

## Splits by speaker

- - crema_d: 91 speakers, train 63 / val 14 / test 14 speakers (seed 42, by speaker, no overlap).
- - emodb: speakers 10 split 7/1/2 by speaker (seed 42): train ['08', '09', '11', '13', '14', '15', '16'], val ['12'], test ['03', '10'].
- - emodb: 10 speakers, train 7 / val 1 / test 2 speakers (seed 42, by speaker, no overlap).
- - esd: speakers 10 split 7/1/2 by speaker (seed 42): train ['0012', '0013', '0016', '0017', '0018', '0019', '0020'], val ['0014'], test ['0011', '0015'].
- - esd: 10 speakers, train 7 / val 1 / test 2 speakers (seed 42, by speaker, no overlap).
- - meld: official train/dev/test split (dev = val). The same cast members appear in all splits, so this is NOT speaker-independent.
- - ravdess: 24 speakers, train 16 / val 4 / test 4 speakers (seed 42, by speaker, no overlap).
- - savee: speakers 4 split 2/1/1 by speaker (seed 42): train ['DC', 'JK'], val ['JE'], test ['KL'].
- - savee: 4 speakers, train 2 / val 1 / test 1 speakers (seed 42, by speaker, no overlap).
- - tess: split by word (200 words: train/val/test by 70/15/15, seed 42). Both speakers appear in every split, so this is NOT speaker-independent.

## Dataset x split x label

| dataset | split | angry | disgust | fear | happy | neutral | sad | surprise | total |
|---|---|---|---|---|---|---|---|---|---|
| crema_d | test | 147 | 150 | 145 | 164 | 154 | 155 | 0 | 915 |
| crema_d | train | 697 | 682 | 702 | 686 | 694 | 681 | 0 | 4142 |
| crema_d | val | 156 | 168 | 153 | 150 | 152 | 164 | 0 | 943 |
| emodb | test | 24 | 2 | 12 | 11 | 15 | 10 | 0 | 74 |
| emodb | train | 91 | 42 | 51 | 58 | 60 | 48 | 0 | 350 |
| emodb | val | 12 | 2 | 6 | 2 | 4 | 4 | 0 | 30 |
| esd | test | 213 | 0 | 0 | 245 | 230 | 242 | 232 | 1162 |
| esd | train | 863 | 0 | 0 | 839 | 842 | 841 | 857 | 4242 |
| esd | val | 124 | 0 | 0 | 116 | 128 | 117 | 111 | 596 |
| meld | test | 230 | 68 | 50 | 186 | 233 | 208 | 197 | 1172 |
| meld | train | 737 | 271 | 268 | 820 | 764 | 683 | 786 | 4329 |
| meld | val | 102 | 22 | 40 | 64 | 73 | 111 | 87 | 499 |
| ravdess | test | 32 | 32 | 32 | 32 | 48 | 32 | 32 | 240 |
| ravdess | train | 128 | 128 | 128 | 128 | 192 | 128 | 128 | 960 |
| ravdess | val | 32 | 32 | 32 | 32 | 48 | 32 | 32 | 240 |
| savee | test | 15 | 15 | 15 | 15 | 30 | 15 | 15 | 120 |
| savee | train | 30 | 30 | 30 | 30 | 60 | 30 | 30 | 240 |
| savee | val | 15 | 15 | 15 | 15 | 30 | 15 | 15 | 120 |
| tess | test | 60 | 60 | 60 | 60 | 60 | 60 | 60 | 420 |
| tess | train | 280 | 280 | 280 | 280 | 280 | 280 | 280 | 1960 |
| tess | val | 60 | 60 | 60 | 60 | 60 | 60 | 60 | 420 |

Overlap check: no speaker appears in two splits (asserted) for ravdess, crema_d, emodb, savee and esd. TESS: no word appears in two splits (asserted); speakers overlap by design. MELD: official split, cast members overlap.

## Colab package

- project\colab_upload_audio_v2\audio_16k.zip: 1640.9 MB
- Notebook: copied
