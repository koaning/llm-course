# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    slider = mo.ui.slider(1, 100, 1, label="a =")
    slider
    return (slider,)


app._unparsable_cell(
    r"""
        a = 1
    """,
    name="_"
)


@app.cell
def _(slider):
    a = slider.value
    return (a,)


@app.cell
def _():
    b = 2
    return (b,)


@app.cell
def _(a, b):
    a + b
    return


if __name__ == "__main__":
    app.run()
