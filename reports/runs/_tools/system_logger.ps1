# trainer-ops: one line every 60 s to <run>/system.csv; exits when the trainer PID is gone
# or at the deadline (HH:mm, local), whichever comes first.
param([Parameter(Mandatory=$true)][int]$TrainerPid, [Parameter(Mandatory=$true)][string]$Out, [Parameter(Mandatory=$true)][string]$Deadline)
$stopAt = [datetime]::ParseExact($Deadline, "HH:mm", $null)
if (-not (Test-Path $Out)) { "time,cpu_total_pct,available_mbytes,vizdoom_count,trainer_alive" | Out-File -Encoding ascii $Out }
while ($true) {
  $alive = $null -ne (Get-Process -Id $TrainerPid -ErrorAction SilentlyContinue)
  $c = (Get-Counter '\Processor(_Total)\% Processor Time','\Memory\Available MBytes').CounterSamples
  $n = @(Get-Process -Name vizdoom -ErrorAction SilentlyContinue).Count
  ("{0},{1:N1},{2},{3},{4}" -f (Get-Date -Format 'yyyy-MM-ddTHH:mm:ss'), $c[0].CookedValue, $c[1].CookedValue, $n, [int]$alive) | Out-File -Append -Encoding ascii $Out
  if (-not $alive -or (Get-Date) -ge $stopAt) { break }
  Start-Sleep -Seconds 59
}
