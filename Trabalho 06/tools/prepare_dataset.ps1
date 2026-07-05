param(
    [string]$BasePath = "..\by_class",
    [string]$OutCsv = "..\dataset_28x28.csv",
    [string]$OutClasses = "..\classes.txt",
    [int]$ImageSize = 28,
    [int]$MaxPerClass = 350,
    [int]$Seed = 42
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Drawing

function Get-ClassEntries {
    param([string]$byClass)

    $dirs = Get-ChildItem -Path $byClass -Directory | Sort-Object Name
    $entries = @()

    foreach ($d in $dirs) {
        if ($d.Name -notmatch '^[0-9a-fA-F]{2}$') { continue }

        $ascii = [Convert]::ToInt32($d.Name, 16)
        if (($ascii -ge 48 -and $ascii -le 57) -or ($ascii -ge 65 -and $ascii -le 90) -or ($ascii -ge 97 -and $ascii -le 122)) {
            $entries += [PSCustomObject]@{
                Hex = $d.Name.ToLower()
                Ascii = $ascii
                Char = [char]$ascii
                Path = $d.FullName
            }
        }
    }

    return $entries | Sort-Object Ascii
}

function To-InkValue {
    param([int]$r, [int]$g, [int]$b)

    $gray = 0.299 * $r + 0.587 * $g + 0.114 * $b
    $ink = 1.0 - ($gray / 255.0)
    if ($ink -lt 0) { $ink = 0 }
    if ($ink -gt 1) { $ink = 1 }
    return $ink
}

function Convert-PngToVector {
    param(
        [string]$imgPath,
        [int]$size
    )

    $src = [System.Drawing.Bitmap]::new($imgPath)
    try {
        $dst = [System.Drawing.Bitmap]::new($size, $size)
        try {
            $g = [System.Drawing.Graphics]::FromImage($dst)
            try {
                $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
                $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
                $g.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
                $g.Clear([System.Drawing.Color]::White)
                $g.DrawImage($src, 0, 0, $size, $size)
            }
            finally {
                $g.Dispose()
            }

            $vals = New-Object System.Collections.Generic.List[string]
            for ($y = 0; $y -lt $size; $y++) {
                for ($x = 0; $x -lt $size; $x++) {
                    $c = $dst.GetPixel($x, $y)
                    $ink = To-InkValue -r $c.R -g $c.G -b $c.B
                    $vals.Add(([string]::Format([Globalization.CultureInfo]::InvariantCulture, "{0:0.000000}", $ink)))
                }
            }

            return $vals
        }
        finally {
            $dst.Dispose()
        }
    }
    finally {
        $src.Dispose()
    }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$baseFull = Resolve-Path (Join-Path $scriptDir $BasePath)
$outCsvFull = Join-Path $scriptDir $OutCsv
$outClsFull = Join-Path $scriptDir $OutClasses

$classes = Get-ClassEntries -byClass $baseFull
if ($classes.Count -eq 0) {
    throw "Nenhuma classe valida encontrada em $baseFull"
}

$rand = [System.Random]::new($Seed)
$rows = New-Object System.Collections.Generic.List[string]

Write-Host "Classes encontradas: $($classes.Count)"

$header = New-Object System.Collections.Generic.List[string]
$header.Add("label")
for ($i = 0; $i -lt ($ImageSize * $ImageSize); $i++) {
    $header.Add("p$i")
}
$rows.Add(($header -join ","))

$classLines = New-Object System.Collections.Generic.List[string]

for ($classIndex = 0; $classIndex -lt $classes.Count; $classIndex++) {
    $c = $classes[$classIndex]
    $trainDir = Join-Path $c.Path ("train_" + $c.Hex)
    if (-not (Test-Path $trainDir)) {
        Write-Warning "Pasta de treino nao encontrada para classe $($c.Hex): $trainDir"
        continue
    }

    $files = Get-ChildItem -Path $trainDir -Filter "*.png" -File
    if ($files.Count -eq 0) {
        Write-Warning "Sem imagens em $trainDir"
        continue
    }

    $files = $files | Sort-Object Name

    if ($files.Count -gt $MaxPerClass) {
        $files = $files | Sort-Object { $rand.Next() } | Select-Object -First $MaxPerClass
    }

    $classLines.Add("$classIndex`t$($c.Hex)`t$($c.Char)")

    $count = 0
    foreach ($f in $files) {
        $vec = Convert-PngToVector -imgPath $f.FullName -size $ImageSize
        $rows.Add(("$classIndex," + ($vec -join ",")))
        $count++
    }

    Write-Host ([string]::Format("Classe {0} ({1}) -> {2} amostras", $c.Char, $c.Hex, $count))
}

Set-Content -Path $outCsvFull -Value $rows -Encoding UTF8
Set-Content -Path $outClsFull -Value $classLines -Encoding UTF8

Write-Host "\n[OK] Dataset gerado: $outCsvFull"
Write-Host "[OK] Mapa de classes: $outClsFull"
