# Build the manuscript PDF. Requires a LaTeX toolchain (pdflatex + bibtex).
MAIN := main

.PHONY: all clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib tables.tex results_macros.tex
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	bibtex $(MAIN) || true
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex

# Regenerate tables.tex + results_macros.tex from the committed metrics, then build.
tables.tex results_macros.tex:
	cd .. && uv run python scripts/make_paper_tables.py

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).bbl $(MAIN).blg \
	      $(MAIN).toc $(MAIN).synctex.gz $(MAIN).fls $(MAIN).fdb_latexmk
