# FDE-Bench project page

This directory contains the static research page for **FDE-Bench: Evaluating End-to-End Delivery from Underspecified Real-World Business Requests**.

- Live page: <https://sponyo531.github.io/fde-bench/>
- Paper: [`assets/FDE-Bench.pdf`](assets/FDE-Bench.pdf)
- Code and case specifications: <https://github.com/sponyo531/fde-bench>

## Preview locally

From the repository root:

```bash
python -m http.server 8000 --directory docs
```

The page has no build step and uses only the self-hosted fonts, figures, data, and scripts in this directory. It has no analytics or CDN dependency.

## Files

- `index.html` — page structure, research copy, resource links, and metadata.
- `styles.css` — responsive layout and visual system.
- `app.js` — leaderboard, ablation, case filters, tabs, menu, and citation interactions.
- `data.js` — reported results, case inventory, authors, and paper-linked study data.
- `assets/` — paper PDF, figures, logo, and self-hosted fonts.

The benchmark release, data-download instructions, license status, and reproducibility checks live in the repository root README and [`DATA.md`](../DATA.md). The page labels Case 040 as a protocol whose row-level HRS data are withheld pending redistribution confirmation.
