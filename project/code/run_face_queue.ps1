# Waits for the CUDA training venv, then trains every face model in sequence.
# Usage (repo root): powershell -File project\code\run_face_queue.ps1
# Logs: project\logs\face_<model>.log ; a model whose metrics JSON exists is skipped, so re-running resumes.
$Root = Resolve-Path "$PSScriptRoot\..\.."
$Py   = Join-Path $Root "project\.venv-train\Scripts\python.exe"
$Logs = Join-Path $Root "project\logs"
New-Item -ItemType Directory -Force $Logs | Out-Null

while ($true) {
    & $Py -c "import torch, sklearn; assert torch.cuda.is_available()" 2>$null
    if ($LASTEXITCODE -eq 0) { break }
    Write-Host "$(Get-Date -Format s) waiting for CUDA torch in .venv-train..."
    Start-Sleep -Seconds 60
}

# Baselines first (they make the app work), then the faculty hybrids, lightest to heaviest.
foreach ($m in "cnn", "resnet18", "mobilenet_attn", "cnn_bilstm", "densenet_se", "effnet_cbam", "resnet50_vit") {
    if (Test-Path (Join-Path $Root "project\reports\face_${m}_metrics.json")) { Write-Host "skip $m (done)"; continue }
    # A finished checkpoint without metrics (run killed during the test step) only needs evaluation.
    $extra = if ($m -eq "resnet18" -and (Test-Path (Join-Path $Root "models\resnet18_fer.pt"))) { "--eval-only" } else { "" }
    Write-Host "$(Get-Date -Format s) training $m $extra"
    # cmd /c: UTF-8 log, and Python stderr warnings don't become PowerShell errors.
    cmd /c "`"$Py`" `"$(Join-Path $Root 'project\code\train_face.py')`" --model $m $extra > `"$Logs\face_$m.log`" 2>&1"
    Write-Host "$(Get-Date -Format s) finished $m (exit $LASTEXITCODE)"
    Get-Content "$Logs\face_$m.log" -Tail 1
}
Write-Host "$(Get-Date -Format s) face queue complete"
