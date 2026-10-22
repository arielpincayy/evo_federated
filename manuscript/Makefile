# Build the PDF inside the TeX Live container (no local LaTeX needed).
pdf:
	docker run --rm -v $(CURDIR):/w -w /w texlive/texlive:latest latexmk -pdf -interaction=nonstopmode main.tex
informe:
	docker run --rm -v $(CURDIR):/w -w /w texlive/texlive:latest sh -c "latexmk -pdf -interaction=nonstopmode informe.tex"
clean:
	docker run --rm -v $(CURDIR):/w -w /w texlive/texlive:latest latexmk -C
