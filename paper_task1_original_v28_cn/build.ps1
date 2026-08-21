$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

Push-Location $PSScriptRoot
try {
    if (-not (Get-Command xelatex -ErrorAction SilentlyContinue)) {
        throw "未找到 xelatex。请安装 TeX Live 或 MiKTeX，并确保 xelatex 在 PATH 中。"
    }
    if (-not (Get-Command bibtex -ErrorAction SilentlyContinue)) {
        throw "未找到 bibtex。请安装 TeX Live 或 MiKTeX，并确保 bibtex 在 PATH 中。"
    }

    & xelatex -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) { throw "第一次 XeLaTeX 编译失败。" }
    & bibtex main
    if ($LASTEXITCODE -ne 0) { throw "BibTeX 编译失败。" }
    & xelatex -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) { throw "第二次 XeLaTeX 编译失败。" }
    & xelatex -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) { throw "第三次 XeLaTeX 编译失败。" }

    Write-Host "完成：$PSScriptRoot\main.pdf"
}
finally {
    Pop-Location
}

