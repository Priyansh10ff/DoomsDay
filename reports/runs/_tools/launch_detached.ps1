# trainer-ops: start configs.ops_launch detached (survives the session), stdout/stderr to the run dir,
# write trainer.pid, then start the system logger detached. Prints the PID.
param([Parameter(Mandatory=$true)][string]$Config, [Parameter(Mandatory=$true)][string]$RunDir, [Parameter(Mandatory=$true)][string]$Deadline, [string]$Resume = "")
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
$pyArgs = @("-u", "-m", "configs.ops_launch", $Config, $RunDir)
if ($Resume) { $pyArgs += @("--resume", $Resume) }
$sfx = if ($Resume) { "_resume_" + (Get-Date -Format 'HHmmss') } else { "" }
$p = Start-Process -FilePath "python" -ArgumentList $pyArgs -WorkingDirectory "D:\DoomsDay" -WindowStyle Hidden -PassThru `
     -RedirectStandardOutput (Join-Path $RunDir "stdout$sfx.log") -RedirectStandardError (Join-Path $RunDir "stderr$sfx.log")
"$($p.Id)" | Out-File -Encoding ascii (Join-Path $RunDir "trainer$sfx.pid")
$lg = Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile","-ExecutionPolicy","Bypass","-File","D:\DoomsDay\reports\runs\_tools\system_logger.ps1","-TrainerPid",$p.Id,"-Out",(Join-Path $RunDir "system.csv"),"-Deadline",$Deadline) -WindowStyle Hidden -PassThru
"$($lg.Id)" | Out-File -Encoding ascii (Join-Path $RunDir "logger$sfx.pid")
"trainer_pid=$($p.Id) logger_pid=$($lg.Id)"
