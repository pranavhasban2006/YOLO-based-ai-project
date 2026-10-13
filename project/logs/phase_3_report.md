# Phase 3 report

Rows written: 85318 -> project/data/splits.csv

## Rows per dataset and split

```
split    test  train   val
dataset                   
crema_d  5710  25765  5730
fer2013  6722  24100  2678
jaffe      44    128    41
ravdess  2400   9600  2400
```

## Subjects per dataset and split (video and jaffe)

```
split    test  train  val
dataset                  
crema_d    14     63   14
jaffe       2      6    2
ravdess     4     16    4
```

fer2013 rows dropped because label_ferplus is empty: 2387

## Class counts in the combined train split

```
angry        7857
disgust      5856
fear         6282
happy       12367
neutral     14131
sad          8762
surprise     4338
```

## Class weights (saved to project/data/class_weights.json)

```
angry       0.934
disgust     1.254
fear        1.169
happy       0.594
neutral     0.520
sad         0.838
surprise    1.692
```

Checks passed: no subject in two splits within a dataset; no face_path twice.
