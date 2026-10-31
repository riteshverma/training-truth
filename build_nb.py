"""Convert lab.py (py:percent cells) into training_truth.ipynb."""
import re
import nbformat

src = open("lab.py", encoding="utf-8").read()
parts = re.split(r"^# %%( \[markdown\])?\n", src, flags=re.M)
cells = []
for i in range(1, len(parts), 2):
    md, body = parts[i], parts[i + 1].rstrip("\n")
    if md:
        cells.append(nbformat.v4.new_markdown_cell(
            "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.split("\n"))))
    else:
        cells.append(nbformat.v4.new_code_cell(body))
nb = nbformat.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbformat.write(nb, "training_truth.ipynb")
