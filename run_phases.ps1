$ErrorActionPreference = "Continue"
$Root      = (Get-Location).Path
$Datasets  = Join-Path $Root "datasets"
$PhasesMd  = Join-Path $PSScriptRoot "face_model_phases.md"
$LogDir    = Join-Path $Root "project\logs"
$StateFile = Join-Path $Root "project\state.json"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$RunnerLog = Join-Path $LogDir "runner.log"
function Log($m) { $l = "$(Get-Date -Format s)  $m"; Write-Host $l; Add-Content $RunnerLog $l }
powercfg /change standby-timeout-ac 0 | Out-Null
powercfg /change hibernate-timeout-ac 0 | Out-Null
function Get-Snapshot {
  $files = Get-ChildItem $Datasets -Recurse -File -ErrorAction SilentlyContinue
  $partial = $files | Where-Object { $_.Extension -in ".part",".crdownload",".tmp",".download" }
  [pscustomobject]@{ Bytes = ($files | Measure-Object Length -Sum).Sum; Partial = @($partial).Count }
}
Log "Waiting for downloads in $Datasets to finish..."
$stable = 0; $last = -1
while ($stable -lt 5) {
  Start-Sleep -Seconds 60
  $s = Get-Snapshot
  if ($s.Partial -eq 0 -and $s.Bytes -eq $last) { $stable++ } else { $stable = 0 }
  $last = $s.Bytes
  Log ("size={0:N0} MB  partial_files={1}  stable_checks={2}/5" -f ($s.Bytes/1MB), $s.Partial, $stable)
}
Log "Downloads look finished. Starting phases."
$md = Get-Content $PhasesMd -Raw
$found = [regex]::Matches($md, '(?s)## Phase (\d):[^\n]*\n\s*```\n(.*?)\n```')
$prompts = @{}
foreach ($m in $found) { $prompts[[int]$m.Groups[1].Value] = $m.Groups[2].Value }
if ($prompts.Count -ne 8) { Log "Expected 8 phases, found $($prompts.Count). Stopping."; exit 1 }
function Get-PhaseDone {
  if (Test-Path $StateFile) { try { return [int](Get-Content $StateFile -Raw | ConvertFrom-Json).phase_done } catch { } }
  return -1
}
for ($n = 0; $n -le 7; $n++) {
  if ((Get-PhaseDone) -ge $n) { Log "Phase $n already done, skipping."; continue }
  if ($n -ge 5 -and -not (Test-Path (Join-Path $Root "project\colab_results"))) {
    Log "Waiting for Colab. See project\COLAB_INSTRUCTIONS.md, then run this script again."
    Write-Host "Waiting for Colab. See project\COLAB_INSTRUCTIONS.md, then run this script again."
    exit 0
  }
  $ok = $false
  foreach ($attempt in 1..2) {
    Log "Phase $n, attempt $attempt starting."
    $out = Join-Path $LogDir "claude_phase_${n}_attempt_$attempt.log"
    $prompts[$n] | claude -p --allowedTools "Bash,Read,Write,Edit,Glob,Grep" 2>&1 | Tee-Object -FilePath $out
    if ((Get-PhaseDone) -ge $n) { $ok = $true; break }
    Log "Phase $n did not mark itself done."
  }
  if (-not $ok) { Log "Phase $n failed twice. Stopping. See $LogDir."; exit 1 }
  Log "Phase $n done."
}
Log "All phases finished. Read project\reports\FACE_MODEL_REPORT.md."
