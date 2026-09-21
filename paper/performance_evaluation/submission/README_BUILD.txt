Performance Evaluation submission sources

Build (elsarticle class, TeX Live 2023 or later):

    pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

main.bbl is included so the bibliography also builds without BibTeX. Figures are vector PDF in figures/.
No non-standard class or style files are needed.
