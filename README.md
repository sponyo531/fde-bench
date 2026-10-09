# FDE-Bench project page

A standalone research project page for **FDE-Bench: Evaluating End-to-End Delivery from Underspecified Real-World Business Requests**.

Project page: <https://sponyo531.github.io/fde-bench/>

The page includes sortable results for 18 agent configurations, a searchable inventory of 49 cases, an illustrative warehouse example, the evaluation protocol, authors, and a downloadable paper. The research code and dataset are not linked or included.

## Preview locally

No build step or Node dependencies are needed. From this directory:

```bash
python -m http.server 8000
```

Open <http://localhost:8000>. If running on a remote machine, forward port 8000 in your editor. All fonts, scripts, figures, and the PDF are served locally; the page does not depend on an external CDN or analytics service.

## GitHub Pages

Use the **contents of this directory** as the root of a dedicated repository, such as `sponyo531/fde-bench`. Do not upload the surrounding research workspace.

1. Push these files to the repository's `main` branch.
2. In **Settings → Pages → Build and deployment**, choose **GitHub Actions**.
3. Run **Deploy project page** from the Actions tab, or push another commit to `main`.
4. GitHub will display the deployed URL in Settings → Pages. For that repository name it will normally be `https://sponyo531.github.io/fde-bench/`.

The included workflow publishes the static files with GitHub's official Pages actions. Relative asset paths work both at a project URL and at a domain root. No custom domain is configured.

## Update content

- `index.html`: page copy, section structure, metadata, and PDF links.
- `styles.css`: layout, type, colors, and mobile rules.
- `app.js`: filters, sorting, case browsing, accessible tabs, and citation copy.
- `data.js`: result rows, the case inventory, authors, and affiliations.
- `assets/FDE-Bench.pdf`: the current manuscript; replace it when the paper changes.
- `assets/logo.png`: the supplied team icon without the original wordmark.
- `assets/fonts/`: self-hosted DM Sans and Source Serif 4, with their SIL Open Font License notices.

Result values and case descriptions were extracted from `benchmark/arxiv/5_experiments.tex` (Table 1) and `benchmark/arxiv/8_appendix.tex` (case-level evaluator reference). Author information comes from `benchmark/arxiv/main.tex`. The dataset is a snapshot, not an automatically updated leaderboard. Quality uses four decimal places; the other metrics are percentages and use two.

Once an arXiv identifier is available, update the primary paper links and the BibTeX entry generated in `app.js`. Do not invent an identifier or link to the previous anonymous repository. If the site moves to another repository or domain, update the absolute `og:image`, canonical, and `og:url` metadata in `index.html`.

## Design and attribution

The original page layout takes editorial inspiration from [RSI-Exam](https://rsi-exam.ai/): serif headlines, restrained color, clear results, and section labels. Its code and assets are not copied. The scientific figures, logo, and paper are supplied project material. Font license notices are kept in `assets/fonts/`; no additional license is imposed here on the research paper, figures, data, or research code.
