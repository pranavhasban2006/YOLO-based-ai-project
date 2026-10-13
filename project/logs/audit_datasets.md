# Dataset audit (2026-10-09T23:27:13)

| dataset | status | used files | zips (unzip -t) | reasons |
|---|---|---|---|---|
| fer2013 | COMPLETE | 35887 | FER-2013\archive (1).zip: ok | - |
| ferplus | COMPLETE | 14 | none | - |
| jaffe | COMPLETE | 214 | jaffe\jaffe.zip: ok | - |
| ravdess | COMPLETE | 4345 | RAVDESS\Audio_Speech\Audio_Speech_Actors_01-24.zip: ok, RAVDESS\Video_Song\Video_Song_Actor_01.zip: ok, RAVDESS\Video_Song\Video_Song_Actor_02.zip: ok, RAVDESS\Video_Song\Video_Song_Actor_03.zip: ok, RAVDESS\Video_Speech\Video_Speech_Actor_01.zip: ok, RAVDESS\Video_Speech\Video_Speech_Actor_02.zip: ok, RAVDESS\Video_Speech\Video_Speech_Actor_03.zip: ok, RAVDESS\Video_Speech\Video_Speech_Actor_04.zip: ok, RAVDESS\Video_Speech\Video_Speech_Actor_05.zip: ok, RAVDESS\Video_Speech\Video_Speech_Actor_06.zip: ok, RAVDESS\Video_Speech\Video_Speech_Actor_07.zip: ok, RAVDESS\Video_Speech\Video_Speech_Actor_08.zip: ok, RAVDESS\Video_Speech\Video_Speech_Actor_09.zip: ok, RAVDESS\Video_Speech\Video_Speech_Actor_10.zip: ok, RAVDESS\Video_Speech\Video_Speech_Actor_11.zip: ok, RAVDESS\Video_Speech\Video_Speech_Actor_12.zip: ok, RAVDESS\Video_Speech\Video_Speech_Actor_13.zip: ok, RAVDESS\Video_Speech\Video_Speech_Actor_14.zip: ok, RAVDESS\Video_Speech\Video_Speech_Actor_15.zip: ok, RAVDESS\Video_Speech\Video_Speech_Actor_16.zip: ok, RAVDESS\Video_Speech\Video_Speech_Actor_17.zip: ok, RAVDESS\Video_Speech\Video_Speech_Actor_18.zip: ok, RAVDESS\Video_Speech\Video_Speech_Actor_19.zip: ok, RAVDESS\Video_Speech\Video_Speech_Actor_20.zip: ok, RAVDESS\Video_Speech\Video_Speech_Actor_21.zip: ok, RAVDESS\Video_Speech\Video_Speech_Actor_22.zip: ok, RAVDESS\Video_Speech\Video_Speech_Actor_23.zip: ok, RAVDESS\Video_Speech\Video_Speech_Actor_24.zip: ok | - |
| crema_d | COMPLETE | 14884 | none | - |

## fer2013 — COMPLETE
- folders: FER-2013
- FER-2013: 35889 files, 0.120 GB, partial=0, zero-byte=0, lfs-pointers=0
- fer2013: 10206 files, 0.016 GB, partial=0, zero-byte=0, lfs-pointers=0
- expected `extracted/train`: found=True files=28709 expected=28709
- expected `extracted/test`: found=True files=7178 expected=7178
- zip `FER-2013\archive (1).zip` 63.3 MB unzip -t: No errors detected in compressed data of D:\Projects\YOLO\datasets\FER-2013\archive (1).zip. extracted=True

## ferplus — COMPLETE
- folders: ferplus
- ferplus: 14 files, 0.004 GB, partial=0, zero-byte=0, lfs-pointers=0
- expected `fer2013new.csv`: found=True files=None expected=None

## jaffe — COMPLETE
- folders: jaffe
- jaffe: 217 files, 0.026 GB, partial=0, zero-byte=0, lfs-pointers=0
- expected `extracted`: found=True files=214 expected=None
- zip `jaffe\jaffe.zip` 12.3 MB unzip -t: No errors detected in compressed data of D:\Projects\YOLO\datasets\jaffe\jaffe.zip. extracted=True

## ravdess — COMPLETE
- folders: RAVDESS
- RAVDESS: 4349 files, 29.302 GB, partial=0, zero-byte=0, lfs-pointers=0
- expected `Audio_Speech`: found=True files=1441 expected=None
- expected `Video_Speech`: found=True files=2904 expected=None
- zip `RAVDESS\Audio_Speech\Audio_Speech_Actors_01-24.zip` 208.5 MB unzip -t: No errors detected in compressed data of D:\Projects\YOLO\datasets\RAVDESS\Audio_Speech\Audio_Speech_Actors_01-24.zip. extracted=True
- zip `RAVDESS\Video_Song\Video_Song_Actor_01.zip` 502.5 MB unzip -t: No errors detected in compressed data of D:\Projects\YOLO\datasets\RAVDESS\Video_Song\Video_Song_Actor_01.zip. extracted=False
- zip `RAVDESS\Video_Song\Video_Song_Actor_02.zip` 553.4 MB unzip -t: No errors detected in compressed data of D:\Projects\YOLO\datasets\RAVDESS\Video_Song\Video_Song_Actor_02.zip. extracted=False
- zip `RAVDESS\Video_Song\Video_Song_Actor_03.zip` 508.7 MB unzip -t: No errors detected in compressed data of D:\Projects\YOLO\datasets\RAVDESS\Video_Song\Video_Song_Actor_03.zip. extracted=False
- zip `RAVDESS\Video_Speech\Video_Speech_Actor_01.zip` 553.0 MB unzip -t: No errors detected in compressed data of D:\Projects\YOLO\datasets\RAVDESS\Video_Speech\Video_Speech_Actor_01.zip. extracted=True
- zip `RAVDESS\Video_Speech\Video_Speech_Actor_02.zip` 570.7 MB unzip -t: No errors detected in compressed data of D:\Projects\YOLO\datasets\RAVDESS\Video_Speech\Video_Speech_Actor_02.zip. extracted=True
- zip `RAVDESS\Video_Speech\Video_Speech_Actor_03.zip` 567.8 MB unzip -t: No errors detected in compressed data of D:\Projects\YOLO\datasets\RAVDESS\Video_Speech\Video_Speech_Actor_03.zip. extracted=True
- zip `RAVDESS\Video_Speech\Video_Speech_Actor_04.zip` 546.6 MB unzip -t: No errors detected in compressed data of D:\Projects\YOLO\datasets\RAVDESS\Video_Speech\Video_Speech_Actor_04.zip. extracted=True
- zip `RAVDESS\Video_Speech\Video_Speech_Actor_05.zip` 563.0 MB unzip -t: No errors detected in compressed data of D:\Projects\YOLO\datasets\RAVDESS\Video_Speech\Video_Speech_Actor_05.zip. extracted=True
- zip `RAVDESS\Video_Speech\Video_Speech_Actor_06.zip` 568.5 MB unzip -t: No errors detected in compressed data of D:\Projects\YOLO\datasets\RAVDESS\Video_Speech\Video_Speech_Actor_06.zip. extracted=True
- zip `RAVDESS\Video_Speech\Video_Speech_Actor_07.zip` 565.7 MB unzip -t: No errors detected in compressed data of D:\Projects\YOLO\datasets\RAVDESS\Video_Speech\Video_Speech_Actor_07.zip. extracted=True
- zip `RAVDESS\Video_Speech\Video_Speech_Actor_08.zip` 562.6 MB unzip -t: No errors detected in compressed data of D:\Projects\YOLO\datasets\RAVDESS\Video_Speech\Video_Speech_Actor_08.zip. extracted=True
- zip `RAVDESS\Video_Speech\Video_Speech_Actor_09.zip` 523.5 MB unzip -t: No errors detected in compressed data of D:\Projects\YOLO\datasets\RAVDESS\Video_Speech\Video_Speech_Actor_09.zip. extracted=True
- zip `RAVDESS\Video_Speech\Video_Speech_Actor_10.zip` 565.6 MB unzip -t: No errors detected in compressed data of D:\Projects\YOLO\datasets\RAVDESS\Video_Speech\Video_Speech_Actor_10.zip. extracted=True
- zip `RAVDESS\Video_Speech\Video_Speech_Actor_11.zip` 518.6 MB unzip -t: No errors detected in compressed data of D:\Projects\YOLO\datasets\RAVDESS\Video_Speech\Video_Speech_Actor_11.zip. extracted=True
- zip `RAVDESS\Video_Speech\Video_Speech_Actor_12.zip` 557.9 MB unzip -t: No errors detected in compressed data of D:\Projects\YOLO\datasets\RAVDESS\Video_Speech\Video_Speech_Actor_12.zip. extracted=True
- zip `RAVDESS\Video_Speech\Video_Speech_Actor_13.zip` 501.9 MB unzip -t: No errors detected in compressed data of D:\Projects\YOLO\datasets\RAVDESS\Video_Speech\Video_Speech_Actor_13.zip. extracted=True
- zip `RAVDESS\Video_Speech\Video_Speech_Actor_14.zip` 552.5 MB unzip -t: No errors detected in compressed data of D:\Projects\YOLO\datasets\RAVDESS\Video_Speech\Video_Speech_Actor_14.zip. extracted=True
- zip `RAVDESS\Video_Speech\Video_Speech_Actor_15.zip` 525.2 MB unzip -t: No errors detected in compressed data of D:\Projects\YOLO\datasets\RAVDESS\Video_Speech\Video_Speech_Actor_15.zip. extracted=True
- zip `RAVDESS\Video_Speech\Video_Speech_Actor_16.zip` 562.4 MB unzip -t: No errors detected in compressed data of D:\Projects\YOLO\datasets\RAVDESS\Video_Speech\Video_Speech_Actor_16.zip. extracted=True
- zip `RAVDESS\Video_Speech\Video_Speech_Actor_17.zip` 551.7 MB unzip -t: No errors detected in compressed data of D:\Projects\YOLO\datasets\RAVDESS\Video_Speech\Video_Speech_Actor_17.zip. extracted=True
- zip `RAVDESS\Video_Speech\Video_Speech_Actor_18.zip` 565.7 MB unzip -t: No errors detected in compressed data of D:\Projects\YOLO\datasets\RAVDESS\Video_Speech\Video_Speech_Actor_18.zip. extracted=True
- zip `RAVDESS\Video_Speech\Video_Speech_Actor_19.zip` 581.1 MB unzip -t: No errors detected in compressed data of D:\Projects\YOLO\datasets\RAVDESS\Video_Speech\Video_Speech_Actor_19.zip. extracted=True
- zip `RAVDESS\Video_Speech\Video_Speech_Actor_20.zip` 561.3 MB unzip -t: No errors detected in compressed data of D:\Projects\YOLO\datasets\RAVDESS\Video_Speech\Video_Speech_Actor_20.zip. extracted=True
- zip `RAVDESS\Video_Speech\Video_Speech_Actor_21.zip` 590.3 MB unzip -t: No errors detected in compressed data of D:\Projects\YOLO\datasets\RAVDESS\Video_Speech\Video_Speech_Actor_21.zip. extracted=True
- zip `RAVDESS\Video_Speech\Video_Speech_Actor_22.zip` 560.9 MB unzip -t: No errors detected in compressed data of D:\Projects\YOLO\datasets\RAVDESS\Video_Speech\Video_Speech_Actor_22.zip. extracted=True
- zip `RAVDESS\Video_Speech\Video_Speech_Actor_23.zip` 545.2 MB unzip -t: No errors detected in compressed data of D:\Projects\YOLO\datasets\RAVDESS\Video_Speech\Video_Speech_Actor_23.zip. extracted=True
- zip `RAVDESS\Video_Speech\Video_Speech_Actor_24.zip` 595.8 MB unzip -t: No errors detected in compressed data of D:\Projects\YOLO\datasets\RAVDESS\Video_Speech\Video_Speech_Actor_24.zip. extracted=True

## crema_d — COMPLETE
- folders: crema_d
- crema_d: 22342 files, 3.746 GB, partial=0, zero-byte=0, lfs-pointers=0
- expected `AudioWAV`: found=True files=7442 expected=7442
- expected `AudioMP3`: found=True files=7442 expected=7442
