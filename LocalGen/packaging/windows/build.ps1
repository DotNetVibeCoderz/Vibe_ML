<#
.SYNOPSIS
    Builds the LocalGen MSI.

.DESCRIPTION
    Publishes the CLI, the server and the Admin Control self-contained into one staging folder,
    then wraps them in an MSI with WiX.

    The three applications share a folder deliberately: they are built from one solution against
    one runtime, so the framework and the native llama.cpp libraries — which are most of the
    payload — are installed once rather than three times.

.PARAMETER Backend
    llama.cpp backend to bundle: cpu (default) or cuda12. A CUDA package is several hundred
    megabytes larger, so the two are shipped as separate downloads rather than one installer.

.EXAMPLE
    pwsh packaging/windows/build.ps1 -Version 0.1.0 -Backend cuda12
#>
[CmdletBinding()]
param(
    [string]$Version = "0.1.0",

    [ValidateSet("cpu", "cuda12")]
    [string]$Backend = "cpu",

    [string]$Configuration = "Release",

    [string]$Runtime = "win-x64",

    # Where the finished MSI lands. Relative paths resolve against the repository root.
    [string]$OutputDirectory = "artifacts"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "../..")
$staging = Join-Path $repoRoot "artifacts/staging/win-$Backend"
$output = if ([System.IO.Path]::IsPathRooted($OutputDirectory)) {
    $OutputDirectory
} else {
    Join-Path $repoRoot $OutputDirectory
}

Write-Host "LocalGen $Version ($Backend, $Runtime)" -ForegroundColor Cyan

# A stale staging folder would be packaged along with the new one: WiX harvests whatever is there.
if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }
New-Item -ItemType Directory -Force -Path $staging, $output | Out-Null

$projects = @(
    "src/LocalGen.Cli/LocalGen.Cli.csproj",
    "src/LocalGen.Server/LocalGen.Server.csproj",
    "src/LocalGen.Desktop/LocalGen.Desktop.csproj"
)

foreach ($project in $projects) {
    Write-Host "  publishing $project" -ForegroundColor DarkGray

    dotnet publish (Join-Path $repoRoot $project) `
        --configuration $Configuration `
        --runtime $Runtime `
        --self-contained true `
        --output $staging `
        -p:LlamaBackend=$Backend `
        -p:Version=$Version `
        -p:DebugType=None `
        --nologo --verbosity quiet

    if ($LASTEXITCODE -ne 0) { throw "publish failed for $project" }
}

# Symbols are a debugging aid, not something to ship in an installer, and they are large.
Get-ChildItem $staging -Filter *.pdb -Recurse | Remove-Item -Force

$payloadSize = (Get-ChildItem $staging -Recurse -File | Measure-Object -Property Length -Sum).Sum
Write-Host ("  payload: {0:N0} files, {1:N0} MB" -f `
    (Get-ChildItem $staging -Recurse -File).Count, ($payloadSize / 1MB)) -ForegroundColor DarkGray

# ── Authenticode over the executables, before they are sealed into the MSI ──────────────────
#
# Signing the MSI alone would leave every .exe inside it unsigned, and SmartScreen judges what
# actually runs, not what delivered it.
$certificate = $env:LOCALGEN_WINDOWS_CERT
$certificatePassword = $env:LOCALGEN_WINDOWS_CERT_PASSWORD
$timestampUrl = "http://timestamp.digicert.com"

function Invoke-Signing {
    param([string[]]$Paths)

    if (-not $certificate) { return }

    $signtool = Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin" -Recurse -Filter signtool.exe `
        -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -match "x64" } |
        Sort-Object FullName -Descending |
        Select-Object -First 1

    if (-not $signtool) { throw "LOCALGEN_WINDOWS_CERT is set but signtool.exe was not found. Install the Windows SDK." }

    foreach ($path in $Paths) {
        # /tr timestamps the signature: without it every installed copy stops validating on the
        # day the certificate expires, rather than the day it was signed.
        & $signtool.FullName sign /fd SHA256 /td SHA256 /tr $timestampUrl `
            /f $certificate /p $certificatePassword $path

        if ($LASTEXITCODE -ne 0) { throw "signing failed for $path" }
    }
}

if ($certificate) {
    Write-Host "  signing executables" -ForegroundColor DarkGray
    Invoke-Signing (Get-ChildItem $staging -Include *.exe, *.dll -Recurse |
        Where-Object { $_.Name -like "localgen*" -or $_.Name -like "LocalGen*" } |
        Select-Object -ExpandProperty FullName)
} else {
    Write-Host "  LOCALGEN_WINDOWS_CERT is not set — building an unsigned installer." -ForegroundColor Yellow
}

# ── MSI ─────────────────────────────────────────────────────────────────────────────────────
$msi = Join-Path $output "LocalGen-$Version-$Runtime$(if ($Backend -ne 'cpu') { "-$Backend" }).msi"

Write-Host "  building $([System.IO.Path]::GetFileName($msi))" -ForegroundColor DarkGray

# The tool manifest at the repository root pins the WiX version, so every machine and the release
# workflow build the installer with the same toolchain.
dotnet tool restore --tool-manifest (Join-Path $repoRoot ".config/dotnet-tools.json") | Out-Null

# WiX extensions live outside the manifest, so a fresh machine — a CI runner especially — has to
# be told about the UI extension separately. Adding one already present is a no-op.
dotnet wix extension add -g WixToolset.UI.wixext/6.0.2 | Out-Null

# Built up as variables first: PowerShell would read `-d Name=(expression)` in argument mode as
# two separate arguments, and WiX then reads the second one as another source file.
$authoring = Join-Path $PSScriptRoot "LocalGen.wxs"
$license = Join-Path $PSScriptRoot "License.rtf"

dotnet wix build $authoring `
    -arch x64 `
    -ext WixToolset.UI.wixext `
    -d "ProductVersion=$Version" `
    -d "PublishDir=$staging" `
    -d "LicenseFile=$license" `
    -out $msi

if ($LASTEXITCODE -ne 0) { throw "wix build failed" }

Invoke-Signing @($msi)

$hash = (Get-FileHash $msi -Algorithm SHA256).Hash.ToLowerInvariant()
"$hash  $([System.IO.Path]::GetFileName($msi))" |
    Out-File (Join-Path $output "SHA256SUMS-windows.txt") -Encoding ascii

Write-Host "✓ $msi" -ForegroundColor Green
Write-Host "  sha256 $hash" -ForegroundColor DarkGray
