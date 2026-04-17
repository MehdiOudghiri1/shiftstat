$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root

function Invoke-Tool {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & $FilePath @Arguments
    return $LASTEXITCODE
}

function Test-NoFatalLatexError {
    param(
        [Parameter(Mandatory = $true)]
        [string]$LogPath
    )

    if (-not (Test-Path $LogPath)) {
        return $false
    }

    $log = Get-Content $LogPath -Raw
    if ($log -match "Fatal error occurred") {
        return $false
    }

    return $true
}

function Try-LatexMk {
    $latexmk = Get-Command latexmk -ErrorAction SilentlyContinue
    if (-not $latexmk) {
        return $false
    }

    if (-not (Get-Command perl -ErrorAction SilentlyContinue)) {
        Write-Host "Skipping latexmk because Perl is not available."
        return $false
    }

    Write-Host "Trying latexmk..."
    $code = Invoke-Tool -FilePath $latexmk.Source -Arguments @(
        "-pdf",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "main.tex"
    )

    if ($code -eq 0) {
        return $true
    }

    Write-Warning "latexmk failed (often due to missing Perl on MiKTeX). Falling back to another compiler."
    return $false
}

function Try-Tectonic {
    $tectonic = Get-Command tectonic -ErrorAction SilentlyContinue
    if (-not $tectonic) {
        return $false
    }

    Write-Host "Trying tectonic..."
    $code = Invoke-Tool -FilePath $tectonic.Source -Arguments @("main.tex")
    return ($code -eq 0)
}

function Invoke-PdfLatexBuild {
    $pdflatex = Get-Command pdflatex -ErrorAction SilentlyContinue
    if (-not $pdflatex) {
        return $false
    }

    $bibtex = Get-Command bibtex -ErrorAction SilentlyContinue
    if (-not $bibtex) {
        throw "pdflatex is available but bibtex is missing."
    }

    Write-Host "Trying pdflatex + bibtex..."

    $commonArgs = @(
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        "main.tex"
    )

    $auxPath = Join-Path $PWD "main.aux"
    $pdfPath = Join-Path $PWD "main.pdf"
    $logPath = Join-Path $PWD "main.log"
    $bblPath = Join-Path $PWD "main.bbl"

    $code = Invoke-Tool -FilePath $pdflatex.Source -Arguments $commonArgs
    if ($code -ne 0 -and (-not ((Test-Path $auxPath) -and (Test-NoFatalLatexError -LogPath $logPath)))) {
        throw "The first pdflatex pass failed."
    }

    $code = Invoke-Tool -FilePath $bibtex.Source -Arguments @("main")
    if ($code -ne 0 -and (-not (Test-Path $bblPath))) {
        throw "bibtex failed."
    }

    $code = Invoke-Tool -FilePath $pdflatex.Source -Arguments $commonArgs
    if ($code -ne 0 -and (-not ((Test-Path $pdfPath) -and (Test-NoFatalLatexError -LogPath $logPath)))) {
        throw "The second pdflatex pass failed."
    }

    $code = Invoke-Tool -FilePath $pdflatex.Source -Arguments $commonArgs
    if ($code -ne 0 -and (-not ((Test-Path $pdfPath) -and (Test-NoFatalLatexError -LogPath $logPath)))) {
        throw "The third pdflatex pass failed."
    }

    return $true
}

try {
    if (Try-LatexMk) {
        exit 0
    }

    if (Try-Tectonic) {
        exit 0
    }

    if (Invoke-PdfLatexBuild) {
        exit 0
    }

    throw "No usable LaTeX build path found. Install latexmk with Perl, tectonic, or pdflatex+bibtex."
}
finally {
    Pop-Location
}
