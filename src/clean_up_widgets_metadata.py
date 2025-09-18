# Clean up widgets metadata
from nbformat import read, write, NO_CONVERT

with open("sentiment_dl.ipynb", "r", encoding="utf-8") as f:
    nb = read(f, as_version=4)

# Remove metadata.widgets if present
for cell in nb.cells:
    if "metadata" in cell and "widgets" in cell.metadata:
        del cell.metadata["widgets"]

with open("your_notebook_clean.ipynb", "w", encoding="utf-8") as f:
    write(nb, f)