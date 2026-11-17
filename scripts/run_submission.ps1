param(
    [ValidateSet('Check','Prepare','Baseline','Train','Ensemble','Challenge','Report','All')]
    [string]$Stage = 'Check',
    [string]$PythonExecutable = 'python'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$runLogDirectory = Join-Path $projectRoot 'reports/runs'
New-Item -ItemType Directory -Path $runLogDirectory -Force | Out-Null
$runLogPath = Join-Path $runLogDirectory ("run_{0}_{1}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'), $Stage)
Start-Transcript -Path $runLogPath | Out-Null
function Invoke-ProjectPython {
    param([string[]]$Arguments)
    Write-Host ("Running: {0} {1}" -f $PythonExecutable, ($Arguments -join ' '))
    & $PythonExecutable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Step failed with exit code $LASTEXITCODE. Stop here and follow RUN_AND_REPORT.md." }
}
try {
    $stages = if ($Stage -eq 'All') { @('Prepare','Check','Baseline','Train','Ensemble','Challenge','Report') } else { @($Stage) }
    foreach ($currentStage in $stages) {
        switch ($currentStage) {
            'Prepare' { Invoke-ProjectPython -Arguments @('-m','contextual.preprocessing','--clean-output','data/processed/contextual_clean_v1.jsonl') }
            'Check' { Invoke-ProjectPython -Arguments @('-m','contextual.preflight','--config','configs/contextual_clean_lora.json') }
            'Baseline' { Invoke-ProjectPython -Arguments @('-m','contextual.experiments','--config','configs/contextual_clean_lora.json') }
            'Train' {
                Invoke-ProjectPython -Arguments @('-m','contextual.preflight','--config','configs/contextual_clean_lora.json','--require-gpu')
                Invoke-ProjectPython -Arguments @('-m','contextual.adapter_finetune','--config','configs/contextual_clean_lora.json')
            }
            'Ensemble' { Invoke-ProjectPython -Arguments @('-m','contextual.ensemble','--config','configs/contextual_clean_lora.json','--neural-report','reports/contextual_clean_lora_evaluation.json','--output','reports/clean_ensemble.json') }
            'Challenge' { Invoke-ProjectPython -Arguments @('-m','contextual.recheck','--config','configs/contextual_clean_lora.json','--output','reports/clean_lora_challenge.json') }
            'Report' {
                Invoke-ProjectPython -Arguments @('-m','contextual.statistics')
                Invoke-ProjectPython -Arguments @('-m','contextual.submission')
                Invoke-ProjectPython -Arguments @('scripts/collect_run_results.py')
            }
        }
    }
} finally {
    Stop-Transcript | Out-Null
}
