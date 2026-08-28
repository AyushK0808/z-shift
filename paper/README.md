# Paper

IEEE conference paper for z-shift.

## Layout

```text
paper.tex              the paper (IEEEtran, conference mode)
diagrams/*.mmd         Mermaid sources for the six figures
figures/*.png          rendered figures, referenced by paper.tex
render-diagrams.sh     regenerates figures/ from diagrams/
mermaid-config.json    Mermaid theme (serif, print-friendly greys)
puppeteer-config.json  browser path used by mermaid-cli
```

## Building the PDF

`IEEEtran.cls` is not vendored here. Either compile on Overleaf (it ships
IEEEtran), or install a TeX distribution locally and run:

```bash
pdflatex paper.tex && pdflatex paper.tex   # twice, to resolve \ref and \eqref
```

The second pass is required — figure and equation cross-references come out as
`??` after a single run.

## Regenerating the figures

The figures are generated, not hand-drawn. Edit the `.mmd` source, then:

```bash
./render-diagrams.sh
```

This needs Node.js (for `npx`) and a local Chrome. If Chrome is somewhere other
than the default Windows path, update `executablePath` in
`puppeteer-config.json` — use forward slashes.

All three are full-width (`figure*` at `\textwidth`), rendered at roughly
580–660 dpi, and sized so their text lands at 7–8.5 pt once placed:

| figure | covers | text | height |
|---|---|---|---|
| fig1-overview | end-to-end flow | 7.9 pt | 2.9 in |
| fig2-ingestion-reconstruction | ingestion + reconstruction | 7.4 pt | 4.5 in |
| fig3-refinement-delivery | refinement + rigging + delivery | 8.4 pt | 4.6 in |

Figures 2 and 3 are two-band diagrams: a `flowchart TB` holding two `subgraph`
blocks that each declare `direction LR`. **Connect the bands with a
subgraph-to-subgraph edge (`R1 --> R2`), never node-to-node** — an edge that
crosses a subgraph boundary makes Mermaid silently ignore the inner
`direction`, and both bands collapse into one tall column.

If you rewrite a diagram, watch its natural SVG width — a wide diagram dropped
into `\textwidth` shrinks its own text below legibility. Stay under about
1250 px of natural width to hold 7 pt text. Widening a diagram raises its text
size but also flattens it; narrowing does the reverse, so the two constraints
have to be traded off against each other.

Colour is applied with `classDef` per stage — blue for ingestion, amber for the
schema artifact, violet for reconstruction, teal for refinement, pink for
rigging, orange for deliverables, red for rejection paths. Keep those
assignments consistent across figures: the same stage should be the same colour
in all three.
