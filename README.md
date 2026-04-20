# SKIM Project Page

A static project page for SKIM. Single HTML file, three SVG figures, no build step.

## File structure

```
skim-project-page/
├── index.html                        # the page itself
├── figures/
│   ├── 01_motivating_example.svg     # reading/deciding/writing phases
│   ├── 02_method_scoring.svg         # the scoring formula
│   └── 03_sparsity_frontier.svg      # results plot (placeholder numbers)
└── README.md                          # this file
```

## Hosting on GitHub Pages (first-time walkthrough)

Everything here is free. You need a GitHub account.

1. **Create a public repository** on GitHub. A name like `skim` or `skim-project-page`
   is fine. Do not add a README, .gitignore, or license during creation — you will
   upload everything at once.

2. **Upload the files.** On the empty repository page, use "uploading an existing
   file" and drag in the entire contents of this folder:
   - `index.html`
   - the whole `figures/` directory
   - `README.md` (optional but good practice)

   Commit to `main`.

3. **Enable GitHub Pages.** In your repository: Settings → Pages. Under "Build and
   deployment," source should be "Deploy from a branch," branch `main`, folder `/`
   (root). Save.

4. **Wait ~30 seconds.** The same Settings → Pages panel will show a green banner
   with the live URL, something like `https://<yourusername>.github.io/skim/`.
   That URL is your project page.

5. **Iterate.** Edit files directly on GitHub (click the pencil icon), or clone
   the repo locally, edit, commit, push. Every push to `main` redeploys within
   about 30 seconds.

## What to fill in before sharing

- **Authors.** Change `Anonymous Authors` in the header.
- **Links.** The `Paper / Code / Models` buttons link to `#`. Point them at real URLs.
- **Numbers.** The results table is placeholders (`XX.X`). See the prompts in the
  conversation transcript for how to extract real numbers from your experiments.
- **Citation.** Update the BibTeX block with the real paper info.
- **Title details.** The acronym expansion is currently "Structured Knowledge-aware
  Importance Masking." If the paper uses a different expansion, update it in both
  the `<title>` tag and the `<h1>`.

## Local preview

Open `index.html` in any browser. No server needed. All paths are relative.

## Editing figures

The SVGs are hand-built and render crisp at any size. If you want to edit them,
open them in any SVG editor (Inkscape, Figma, Illustrator) or edit the XML
directly in a text editor — the structure is simple.

If you want to regenerate the sparsity-frontier figure from real data, the curve
coordinates are in `figures/03_sparsity_frontier.svg`. Each data point is a
`<circle>` with `cx`, `cy` attributes, and the line connecting them is a
`<polyline>` with a `points` attribute listing all `x,y` pairs in order. The
y-axis range is 40 (y=380) to 100 (y=140); each y-unit corresponds to 4px.
