# LaTeX Preamble

This file contains LaTeX packages and commands injected into the document
compilation process.

> **Infrastructure note.** This file is parsed by
> `infrastructure/rendering/latex_utils.py` and combined with the configuration
> emitted by `infrastructure/rendering/pdf_renderer.py` before the final Pandoc
> call. The LaTeX below **must** stay inside the fenced code block opened with
> three backticks and the word `latex`: an unfenced preamble is silently dropped,
> which yields a document with no theorem environments, no `cleveref`, and no
> error message.

```latex
% ── Core mathematics ─────────────────────────────────────────────────────────
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsfonts}
\usepackage{amsthm}

% Theorem-like environments. Numbering is delegated to LaTeX — never hand-number
% a definition or proposition in the Markdown source.
\theoremstyle{definition}
\newtheorem{definition}{Definition}[section]
\theoremstyle{plain}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{proposition}{Proposition}[section]

% ── Document layout ──────────────────────────────────────────────────────────
\usepackage{geometry}
% Article dimensions and type are generated from config.yaml into the resolved
% tree. The fixed include survives the provider refreshing this authored file.
\raggedbottom
\setlength{\emergencystretch}{3em}
\setlength{\tabcolsep}{3pt}
\usepackage{float}
\usepackage{graphicx}
\usepackage{pdflscape}
\usepackage{needspace}
\usepackage{tocloft}
\cftsetpnumwidth{3em}
\cftsetrmarg{4em}
\setlength{\cftsecnumwidth}{2em}
\setlength{\cftsubsecnumwidth}{3em}
\setlength{\cftsubsubsecnumwidth}{4.5em}
% Compact only contents spacing and section-entry type; retain every entry.
\setlength{\cftbeforesecskip}{0.5em}
\renewcommand{\cftsecafterpnum}{\vskip -1pt}
\renewcommand{\cftsubsecafterpnum}{\vskip -1pt}
\renewcommand{\cftsubsubsecafterpnum}{\vskip -1pt}

% Float placement. Pandoc's template emits \def\fps@figure{htbp}; `h` (HERE) is not
% governed by \topfraction or \bottomfraction, so a figure that does not fit is
% planted at the current point and runs off the page instead of being deferred.
% Dropping `h` restores LaTeX's normal defer-or-float-page behavior. Do not
% restore it, and do not try to fix an overfull figure with the float fractions.
\makeatletter
\def\fps@figure{tbp}
\makeatother
% ── Tables ───────────────────────────────────────────────────────────────────
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{listings}
\usepackage{fvextra}
\fvset{breaklines=true,breakanywhere=true}
\usepackage{etoolbox}

% Environments used by the formal definitions and reproducibility examples.
\theoremstyle{remark}
\newtheorem{remark}{Remark}[section]
\newtheorem{example}{Example}[section]
\AtBeginEnvironment{longtable}{\small}
\setlength{\LTpre}{\baselineskip}
\setlength{\LTpost}{0pt}

% ── Typography ───────────────────────────────────────────────────────────────
\usepackage{microtype}
% The included style applies to articles; Beamer retains its own geometry/type.
\input{../manuscript/article_typography.sty}
\usepackage{xcolor}
\sloppy
% Keep paragraph starts and ends together in the article output.
\csname @ifclassloaded\endcsname{beamer}{}{\clubpenalty=3000\widowpenalty=10000\brokenpenalty=1000}

% ── Cross-references and citations ───────────────────────────────────────────
% Load order is load-bearing: hyperref before cleveref; preserve this tested package order.
\usepackage{hyperref}
\hypersetup{
    colorlinks=true,
    allcolors=red
}
\usepackage[capitalise,noabbrev]{cleveref}
\usepackage{natbib}
```

## Notation policy

The manuscript deliberately defines **no custom math macros**. Matrix notation is
written out in full (`\mathbf{A}`, `D_{\mathrm{KL}}`,
`\operatorname*{arg\,max}`) so that every equation renders identically through
the LaTeX path and through any HTML or EPUB path, where `header-includes` is not
consulted. A `\newcommand` here would silently produce blank or broken math in
the non-PDF outputs.
