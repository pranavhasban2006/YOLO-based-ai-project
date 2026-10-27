# After the face queue finishes, prepares and trains every WESAD, audio and text model in sequence (one GPU job at a time).
# Usage (repo root): powershell -File project\code\run_rest_queue.ps1
# A job whose metrics JSON exists is skipped, so re-running resumes. Logs: project\logs\<job>.log
$Root = Resolve-Path "$PSScriptRoot\..\.."
$Py   = Join-Path $Root "project\.venv-train\Scripts\python.exe"
$Code = Join-Path $Root "project\code"
$Rep  = Join-Path $Root "project\reports"
$Logs = Join-Path $Root "project\logs"
New-Item -ItemType Directory -Force $Logs | Out-Null

function Run($log, [string[]]$argv) {
    Write-Host "$(Get-Date -Format s) start $log"
    # cmd /c keeps Python's stderr warnings from becoming PowerShell errors; the log stays UTF-8 for grep.
    cmd /c "`"$Py`" $($argv -join ' ') > `"$Logs\$log.log`" 2>&1"
    Write-Host "$(Get-Date -Format s) finished $log (exit $LASTEXITCODE)"
    Get-Content "$Logs\$log.log" -Tail 1
}

while (Get-CimInstance Win32_Process -Filter "name='powershell.exe'" | Where-Object CommandLine -match "run_face_queue") {
    Start-Sleep -Seconds 60
}

if (-not (Test-Path (Join-Path $Root "project\data\wesad_windows.npz"))) { Run "wesad_prepare" @("`"$Code\train_wesad.py`"", "--prepare") }
foreach ($m in "cnn_lstm", "cnn_bilstm", "cnn_gru", "ae_mlp", "transformer_bilstm") {
    if (-not (Test-Path "$Rep\wesad_${m}_metrics.json")) { Run "wesad_$m" @("`"$Code\train_wesad.py`"", "--model", $m) }
}

if (-not (Test-Path (Join-Path $Root "project\data\audio\audio_16k.npz"))) { Run "audio_prepare" @("`"$Code\train_audio.py`"", "--prepare") }
foreach ($m in "cnn_bilstm_attn", "cnn_gru", "wav2vec2_bilstm_attn", "hubert_transformer", "ast_mlp") {
    if (-not (Test-Path "$Rep\audio_${m}_metrics.json")) { Run "audio_$m" @("`"$Code\train_audio.py`"", "--model", $m) }
}

foreach ($d in "goemotions", "emotion") {
    foreach ($m in "distilbert_cnn", "bert_bilstm_attn", "bert_cnn_bilstm", "roberta_bilstm_attn", "deberta_attn") {
        if (-not (Test-Path "$Rep\text_${m}_${d}_metrics.json")) { Run "text_${m}_$d" @("`"$Code\train_text.py`"", "--model", $m, "--dataset", $d) }
    }
}
Write-Host "$(Get-Date -Format s) rest queue complete"
