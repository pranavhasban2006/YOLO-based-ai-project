FACULTY FILES - Multimodal Emotion Recognition using Hybrid Deep Learning Models
Team: Prakul Jain (24BIT0548), Pranav Hasban (24BIT0537), Shreya Mehrotra (24BIT0547)

1_CSV_files
  a_dataset_csv   The datasets in CSV form.
                  fer2013_sample_2_per_class.csv  original FER-2013 layout: emotion, pixels (48x48 = 2304 values), Usage
                  fer2013_class_counts.csv        images per class (train / test)
                  goemotions_ekman7_*.csv         GoEmotions mapped to 7 classes (text, label, emotion)
                  dair_ai_emotion_*.csv           dair-ai Emotion, 6 classes
                  tweeteval_emotion_*.csv         TweetEval emotion, 4 classes
                  speech_ravdess_tess_savee_files.csv   every speech clip: dataset, emotion, file, duration
                  wesad_window_features_32.csv    WESAD: 32 statistical features (mean/std/min/max of 8 chest
                                                  channels) per 60 s window + subject + label
  b_feature_csv   The CSV files used by the SMOTE / 80:20 / 70:30 / 10-fold protocol: one row per held-out sample,
                  columns f0..fN = deep features learned by the hybrid model, label = class. Best model of each
                  modality; PREVIEW_*.csv shows 2 rows per class and the first 10 feature columns.
                  (CSV files of all 27 models: project/features/)
  c_results_csv   protocol_summary_all_models.csv   accuracy / macro-F1 for 80:20, 70:30, 10-fold, with and without SMOTE
                  test_results_all_models.csv       accuracy, precision, recall, F1, CPU/GPU time of every model
                  confusion_matrices_all_models.csv confusion matrix of every model

2_Image_files
  a_dataset_samples  real sample images, 2 per class: FER-2013, CK+, RAVDESS video frames
  b_figures          all charts and diagrams: dataset samples, class balance before/after SMOTE, architecture
                     diagrams, flowchart, comparison charts, confusion matrices, inference time
