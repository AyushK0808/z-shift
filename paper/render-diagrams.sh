#!/usr/bin/env bash
# Renders every Mermaid source in diagrams/ (fig1-4) to a print-resolution PNG in figures/.
#
# Requires Node.js (for npx) and a local Chrome/Chromium at the path given in
# puppeteer-config.json -- adjust that path if Chrome lives elsewhere.
# Re-run this after editing any .mmd file, then rebuild the PDF.
set -euo pipefail
cd "$(dirname "$0")"

for src in diagrams/*.mmd; do
  name="$(basename "$src" .mmd)"
  echo "rendering ${name}"
  npx -y @mermaid-js/mermaid-cli@11 \
    -i "$src" \
    -o "figures/${name}.png" \
    -c mermaid-config.json \
    -p puppeteer-config.json \
    -b white \
    -w 1800 \
    -s 4
done
echo "done: $(ls -1 figures/*.png | wc -l) figures"
