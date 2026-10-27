# Faculty protocol stage: once all training queues finish, export each trained model's features to CSV and run
# protocol.py (80:20 / 70:30 with and without SMOTE, 10-fold CV with and without SMOTE) on it, then summarise.
# Usage (repo root): powershell -File project\code\run_protocol_queue.ps1
# A model whose protocol JSON exists is skipped, so re-running resumes. Logs: project\logs\protocol_<name>.log
$Root = Resolve-Path "$PSScriptRoot\..\.."
$Py   = Join-Path $Root "project\.venv-train\Scripts\python.exe"
$Code = Join-Path $Root "project\code"
$Logs = Join-Path $Root "project\logs"
$Out  = Join-Path $Root "project\reports\protocol"

# Run after training: both use RAM, and running them together got jobs killed for low memory.
while (Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -match "run_(face|rest)_queue" -and $_.CommandLine -notmatch "run_protocol_queue" }) {
    Start-Sleep -Seconds 60
}

$text = foreach ($d in "goemotions", "emotion") {
    foreach ($m in "distilbert_cnn", "bert_bilstm_attn", "bert_cnn_bilstm", "roberta_bilstm_attn", "deberta_attn") { "$m@$d" }
}
$jobs = [ordered]@{
    face  = "cnn", "resnet18", "mobilenet_attn", "cnn_bilstm", "densenet_se", "effnet_cbam", "resnet50_vit"
    wesad = "cnn_lstm", "cnn_bilstm", "cnn_gru", "ae_mlp", "transformer_bilstm"
    audio = "cnn_bilstm_attn", "cnn_gru", "wav2vec2_bilstm_attn", "hubert_transformer", "ast_mlp"
    text  = $text
}
foreach ($mod in $jobs.Keys) {
    foreach ($m in $jobs[$mod]) {
        $name = "${mod}_$($m -replace '@', '_')"
        if (Test-Path "$Out\$name.json") { Write-Host "skip $name (done)"; continue }
        Write-Host "$(Get-Date -Format s) protocol $name"
        cmd /c "`"$Py`" `"$Code\extract_features.py`" --modality $mod --model $m > `"$Logs\protocol_$name.log`" 2>&1 && `"$Py`" `"$Code\protocol.py`" `"$Root\project\features\$name.csv`" >> `"$Logs\protocol_$name.log`" 2>&1"
        Write-Host "$(Get-Date -Format s) finished protocol $name (exit $LASTEXITCODE)"
    }
}
# Accuracy boosts reusing the face models: flip TTA, validation-chosen ensemble, stacked-feature CSV -> protocol.
if (-not (Test-Path "$Out\face_stack.json")) {
    Write-Host "$(Get-Date -Format s) face TTA + ensemble + stack"
    cmd /c "`"$Py`" `"$Code\face_ensemble.py`" > `"$Logs\face_ensemble.log`" 2>&1 && `"$Py`" `"$Code\protocol.py`" `"$Root\project\features\face_stack.csv`" > `"$Logs\protocol_face_stack.log`" 2>&1"
    Write-Host "$(Get-Date -Format s) finished face ensemble (exit $LASTEXITCODE)"
}
cmd /c "`"$Py`" `"$Code\protocol.py`" --summarize > `"$Logs\protocol_summary.log`" 2>&1"
Write-Host "$(Get-Date -Format s) protocol queue complete"
