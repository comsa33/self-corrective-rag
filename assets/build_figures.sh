#!/bin/bash
# Build TikZ figures as standalone PNGs for README
# Requires: pdflatex, pdftoppm (from poppler)

set -e
cd "$(dirname "$0")/.."

FIGURES_DIR="paper/figures"
ASSETS_DIR="assets"
TMP_DIR=$(mktemp -d)

# Common preamble for standalone compilation
PREAMBLE='
\documentclass[border=10pt,tikz]{standalone}
\usepackage{amsmath,amssymb}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{groupplots}
\usetikzlibrary{positioning, arrows.meta, calc, shapes.geometric, fit, backgrounds, patterns}
\newcommand{\ours}{\textsc{TARA}}
'

build_figure() {
    local name=$1
    local tex_file="$FIGURES_DIR/${name}.tex"
    local tmp_tex="$TMP_DIR/${name}.tex"

    echo "Building $name..."

    # Extract just the tikzpicture/axis content (strip figure* wrapper)
    cat > "$tmp_tex" << TEXEOF
${PREAMBLE}
\begin{document}
$(sed -n '/\\begin{tikzpicture}/,/\\end{tikzpicture}/p' "$tex_file")
\end{document}
TEXEOF

    # Compile
    (cd "$TMP_DIR" && pdflatex -interaction=nonstopmode "${name}.tex" > /dev/null 2>&1)

    # Convert to PNG (300 DPI)
    if command -v pdftoppm &> /dev/null; then
        pdftoppm -png -r 300 -singlefile "$TMP_DIR/${name}.pdf" "$ASSETS_DIR/${name}"
        echo "  -> $ASSETS_DIR/${name}.png"
    elif command -v sips &> /dev/null; then
        # macOS fallback: convert PDF to PNG via sips
        sips -s format png "$TMP_DIR/${name}.pdf" --out "$ASSETS_DIR/${name}.png" > /dev/null 2>&1
        echo "  -> $ASSETS_DIR/${name}.png (via sips)"
    else
        cp "$TMP_DIR/${name}.pdf" "$ASSETS_DIR/${name}.pdf"
        echo "  -> $ASSETS_DIR/${name}.pdf (no PNG converter found, install poppler)"
    fi
}

# Build all figures
build_figure "architecture"
build_figure "rq1_f1_comparison"
build_figure "pipeline_comparison"
build_figure "react_example"
build_figure "tool_usage_heatmap"

# Cleanup
rm -rf "$TMP_DIR"
echo "Done!"
