# Makefile for Paper 3 — Mondrian Conformal Prediction (COPA 2026 / PMLR)

PAPER = paper3_mondrian
LATEX = pdflatex
BIBTEX = bibtex

.PHONY: all clean view

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib
	$(LATEX) $(PAPER)
	$(BIBTEX) $(PAPER)
	$(LATEX) $(PAPER)
	$(LATEX) $(PAPER)

clean:
	rm -f $(PAPER).{aux,bbl,blg,log,out,pdf,run.xml,bcf,fls,fdb_latexmk,synctex.gz}

view: $(PAPER).pdf
	xdg-open $(PAPER).pdf 2>/dev/null || open $(PAPER).pdf 2>/dev/null || echo "Open $(PAPER).pdf manually"
