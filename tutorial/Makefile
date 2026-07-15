PYTHON ?= .venv/bin/python
LATEXMK ?= latexmk

.PHONY: pdf code clean

pdf:
	mkdir -p build
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex

code:
	cd .. && for script in tutorial/code/step*.py; do $(PYTHON) "$$script"; done

clean:
	rm -rf build
