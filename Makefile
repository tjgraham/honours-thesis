all: honours-thesis/main.tex
	latexmk -pdf -cd "honours-thesis/main.tex"
	mv "honours-thesis/main.pdf" "honours-thesis.pdf"

clean:
	latexmk -C -cd "honours-thesis/main.tex"
