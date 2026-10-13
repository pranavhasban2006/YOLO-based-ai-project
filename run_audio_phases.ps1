# run_audio_phases.ps1 - waits for the dataset downloads to finish, then runs audio
# Phases 0-3 with Claude Code headless. Phases 4-5 need the Colab results, so they are
# run by hand later. Run from the folder that contains datasets\ and project\.
#   powershell -ExecutionPolicy Bypass -File .\run_audio_phases.ps1
# audio_model_phases.md must be in the same folder as this script.
$ErrorActionPreference = "Continue"
$Root=(Get-Location).Path; $Datasets=Join-Path $Root "datasets"
$PhasesMd=Join-Path $PSScriptRoot "audio_model_phases.md"
$LogDir=Join-Path $Root "project\logs"; $State=Join-Path $Root "project\audio_state.json"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
function Log($m){ $l="$(Get-Date -Format s)  $m"; Write-Host $l; Add-Content (Join-Path $LogDir "audio_runner.log") $l }
powercfg /change standby-timeout-ac 0 | Out-Null; powercfg /change hibernate-timeout-ac 0 | Out-Null
function Snap { $f=Get-ChildItem $Datasets -Recurse -File -ErrorAction SilentlyContinue
  [pscustomobject]@{Bytes=($f|Measure-Object Length -Sum).Sum; Partial=@($f|Where-Object{$_.Extension -in ".part",".crdownload",".tmp",".download"}).Count} }
Log "Waiting for downloads and extraction to finish..."
$stable=0;$last=-1
while($stable -lt 6){ Start-Sleep 60; $s=Snap
  $rep=Join-Path $Datasets "audio_download_report.md"
  $interim = (Test-Path $rep) -and ((Get-Content $rep -Raw) -match "(?i)interim")
  if($s.Partial -eq 0 -and $s.Bytes -eq $last -and -not $interim){$stable++}else{$stable=0}
  $last=$s.Bytes; Log ("size={0:N0} MB partial={1} interim_report={2} stable={3}/6" -f ($s.Bytes/1MB),$s.Partial,$interim,$stable) }
Log "Datasets look finished. Starting audio phases 0-3."
$md=Get-Content $PhasesMd -Raw
$mm=[regex]::Matches($md,'(?s)## Phase (\d):[^\n]*\n\s*```\n(.*?)\n```')
$p=@{}; foreach($m in $mm){$p[[int]$m.Groups[1].Value]=$m.Groups[2].Value}
if($p.Count -ne 4){Log "Expected 4 phases (0-3), found $($p.Count). Stopping."; exit 1}
function Done { if(Test-Path $State){ try{return [int](Get-Content $State -Raw|ConvertFrom-Json).phase_done}catch{} }; return -1 }
for($n=0;$n -le 3;$n++){
  if((Done) -ge $n){Log "Phase $n already done, skipping.";continue}
  $ok=$false
  foreach($a in 1..2){ Log "Phase $n attempt $a"
    $p[$n] | claude -p --allowedTools "Bash,Read,Write,Edit,Glob,Grep" 2>&1 | Tee-Object -FilePath (Join-Path $LogDir "audio_claude_phase_${n}_attempt_$a.log")
    if((Done) -ge $n){$ok=$true;break} }
  if(-not $ok){Log "Phase $n failed twice. Stopping."; exit 1}
  Log "Phase $n done." }
Log "Phases 0-3 finished. Next: upload project\colab_upload_audio\audio_16k.zip to Drive and run the notebook."
