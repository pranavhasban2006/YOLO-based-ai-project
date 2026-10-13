# Makes a short spoken sample (Windows speech synthesis) used as the fake microphone in the browser test and in
# model_service/tests. Output: client/scripts/.cache/speech.wav (2 s silence, the phrase, 3 s silence; 16 kHz mono)
$here = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$raw = Join-Path $here 'client\scripts\.cache\speech_raw.wav'
$out = Join-Path $here 'client\scripts\.cache\speech.wav'
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.Rate = -1
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$s.SetOutputToWaveFile($raw, $fmt)
$s.Speak('I have been feeling a bit tired lately, and I did not sleep well last night.')
$s.Dispose()
ffmpeg -y -loglevel error -i $raw -af "adelay=2000,apad=pad_dur=3" -ar 16000 -ac 1 $out
Copy-Item $out (Join-Path $here 'model_service\tests\sample_speech.wav') -Force
Get-Item $out | Select-Object Name, Length
