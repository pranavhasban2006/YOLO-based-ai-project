| modality | dataset | hybrid | test samples | A1 alone acc | A1 alone F1 | A1 wrong (fed to A2) | fixed by A2 | A1 + A2 (wrong only) acc | A1 + A2 (wrong only) F1 | gain (points) | A2 on all samples acc | broken when A2 sees all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| speech | RAVDESS+TESS+SAVEE | AST + MLP classifier | 1100 | 0.8955 | 0.8957 | 115 | 35 | 0.9273 | 0.9271 | 3.18 | 0.9045 | 25 |
| speech | RAVDESS+TESS+SAVEE | CNN + BiLSTM + Attention | 1100 | 0.84 | 0.8446 | 176 | 93 | 0.9245 | 0.9247 | 8.45 | 0.8991 | 28 |
| speech | RAVDESS+TESS+SAVEE | CNN + GRU | 1100 | 0.84 | 0.8446 | 176 | 99 | 0.93 | 0.9302 | 9.0 | 0.9009 | 32 |
| speech | RAVDESS+TESS+SAVEE | HuBERT + Transformer encoder | 1100 | 0.8809 | 0.8802 | 131 | 72 | 0.9464 | None | 6.55 | 0.9291 | 19 |
| speech | RAVDESS+TESS+SAVEE | wav2vec 2.0 + BiLSTM + Attention | 1100 | 0.7545 | 0.7527 | 270 | 183 | 0.9209 | 0.9209 | 16.64 | 0.9018 | 21 |
| face | FER-2013 | CNN (ResNet-18) + BiLSTM + Attention | 7178 | 0.6936 | 0.6581 | 2199 | 165 | 0.7166 | 0.7151 | 2.3 | 0.6999 | 120 |
| face | FER-2013 | DenseNet121 + SE attention | 7178 | 0.707 | 0.6851 | 2103 | 148 | 0.7276 | 0.7185 | 2.06 | 0.7095 | 130 |
| face | FER-2013 | EfficientNetB3 + CBAM attention | 7178 | 0.7038 | 0.684 | 2126 | 195 | 0.731 | 0.7208 | 2.72 | 0.7041 | 193 |
| face | FER-2013 | MobileNetV3 + Spatial attention | 7178 | 0.6955 | 0.6709 | 2186 | 145 | 0.7157 | 0.7147 | 2.02 | 0.6918 | 171 |
| face | FER-2013 | ResNet50 + Vision Transformer | 7178 | 0.6911 | 0.6573 | 2217 | 240 | 0.7246 | 0.719 | 3.35 | 0.6963 | 203 |
| text | emotion | BERT + BiLSTM + Attention | 2000 | 0.9295 | 0.8874 | 141 | 24 | 0.9415 | 0.9123 | 1.2 | 0.9255 | 32 |
| text | goemotions | BERT + BiLSTM + Attention | 4968 | 0.7087 | 0.6497 | 1447 | 87 | 0.7262 | 0.6666 | 1.75 | 0.7011 | 125 |
| text | emotion | BERT + CNN + BiLSTM | 2000 | 0.9295 | 0.8874 | 141 | 22 | 0.9405 | 0.913 | 1.1 | 0.924 | 33 |
| text | goemotions | BERT + CNN + BiLSTM | 4968 | 0.7087 | 0.6497 | 1447 | 111 | 0.7311 | 0.6708 | 2.24 | 0.6971 | 169 |
| text | emotion | DeBERTa + Attention classifier | 2000 | 0.9295 | 0.8896 | 141 | 30 | 0.9445 | 0.9202 | 1.5 | 0.9325 | 24 |
| text | goemotions | DeBERTa + Attention classifier | 4968 | 0.7013 | 0.654 | 1484 | 141 | 0.7297 | 0.6747 | 2.84 | 0.7053 | 121 |
| text | emotion | DistilBERT + CNN | 2000 | 0.9335 | 0.8886 | 133 | 36 | 0.9515 | 0.926 | 1.8 | 0.9285 | 46 |
| text | goemotions | DistilBERT + CNN | 4968 | 0.7019 | 0.6356 | 1481 | 84 | 0.7188 | 0.6595 | 1.69 | 0.6975 | 106 |
| text | emotion | RoBERTa + BiLSTM + Attention | 2000 | 0.9275 | 0.8858 | 145 | 7 | 0.931 | 0.8938 | 0.35 | 0.9255 | 11 |
| text | goemotions | RoBERTa + BiLSTM + Attention | 4968 | 0.6781 | 0.6298 | 1599 | 464 | 0.7715 | 0.6996 | 9.34 | 0.7057 | 327 |
