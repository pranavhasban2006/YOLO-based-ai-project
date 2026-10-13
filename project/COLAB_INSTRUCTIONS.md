# Train on Google Colab

1) Upload the project/colab_upload folder to Google Drive at MyDrive/YOLO/colab_upload.
2) Open train_colab.ipynb in Google Colab (File > Upload notebook, or open it from Drive).
3) Choose Runtime > Change runtime type > T4 GPU.
4) Choose Runtime > Run all. If Colab disconnects, reconnect and run all again; it resumes.
5) When it prints ALL DONE, copy MyDrive/YOLO/colab_results/ back into project/colab_results/ on this PC.

Expected time on a T4: about 4 to 5 hours with the default MAX_HOURS of 6. If the E3 time estimate
would go past the budget, the remaining held-out runs are skipped and listed in the output.
