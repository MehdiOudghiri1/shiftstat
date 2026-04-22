"""Render README equations from LaTeX to SVG assets.

The generated SVGs are meant for PyPI, where README math is rendered as plain
Markdown/HTML rather than MathJax.  The script intentionally uses the local TeX
toolchain plus dvisvgm so the glyphs come from real LaTeX math fonts.
"""

from __future__ import annotations

import dataclasses
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "assets" / "equations"


@dataclasses.dataclass(frozen=True)
class Equation:
    filename: str
    tex: str


EQUATIONS: tuple[Equation, ...] = (
    Equation(
        "target-risk.svg",
        r"R_{\mathrm{tgt}}(f)"
        r"=\mathbb{E}_{(X,Y)\sim P_{\mathrm{tgt}}}"
        r"\!\left[\ell\!\left(Y,f(X)\right)\right]",
    ),
    Equation(
        "reference-risk-estimator.svg",
        r"\frac{1}{n}\sum_{i=1}^{n}\ell\!\left(y_i,f(x_i)\right)",
    ),
    Equation(
        "covariate-shift-assumption.svg",
        r"P_{\mathrm{ref}}(Y\mid X)=P_{\mathrm{tgt}}(Y\mid X),"
        r"\qquad P_{\mathrm{ref}}(X)\ne P_{\mathrm{tgt}}(X)",
    ),
    Equation(
        "density-ratio.svg",
        r"w(x)=\frac{p_{\mathrm{tgt}}(x)}{p_{\mathrm{ref}}(x)}",
    ),
    Equation(
        "weighted-target-risk.svg",
        r"R_{\mathrm{tgt}}(f)"
        r"=\mathbb{E}_{P_{\mathrm{ref}}}"
        r"\!\left[w(X)\,\ell\!\left(Y,f(X)\right)\right]",
    ),
    Equation(
        "domain-posterior.svg",
        r"d(x)=P(D=\mathrm{target}\mid X=x)",
    ),
    Equation(
        "domain-density-ratio.svg",
        r"w(x)=\frac{d(x)}{1-d(x)}\cdot"
        r"\frac{\pi_{\mathrm{ref}}}{\pi_{\mathrm{tgt}}}",
    ),
    Equation(
        "weighted-empirical-risk.svg",
        r"\widehat{R}_{w}(f)"
        r"=\frac{\sum_{i=1}^{n}w_i\,\ell\!\left(y_i,f(x_i)\right)}"
        r"{\sum_{i=1}^{n}w_i}",
    ),
    Equation(
        "kish-effective-sample-size.svg",
        r"n_{\mathrm{eff}}"
        r"=\frac{\left(\sum_{i=1}^{n}w_i\right)^2}"
        r"{\sum_{i=1}^{n}w_i^2}",
    ),
    Equation(
        "ece.svg",
        r"\mathrm{ECE}"
        r"=\sum_{k=1}^{K}\frac{|B_k|}{n}"
        r"\left|"
        r"\operatorname{mean}_{i\in B_k}(p_i)"
        r"-\operatorname{mean}_{i\in B_k}(y_i)"
        r"\right|",
    ),
    Equation(
        "weighted-ece.svg",
        r"\mathrm{ECE}_{w}"
        r"=\sum_{k=1}^{K}\frac{W_k}{W}"
        r"\left|"
        r"\operatorname{avg}_{w}(p_i\mid i\in B_k)"
        r"-\operatorname{avg}_{w}(y_i\mid i\in B_k)"
        r"\right|",
    ),
    Equation(
        "selective-risk.svg",
        r"R_{\mathrm{tgt}}(f,\phi)"
        r"=\frac{\mathbb{E}_{\mathrm{tgt}}"
        r"\!\left[\phi(X)\,\ell\!\left(Y,f(X)\right)\right]}"
        r"{\mathbb{E}_{\mathrm{tgt}}\!\left[\phi(X)\right]}",
    ),
    Equation(
        "acceptance-function.svg",
        r"\phi(x)\in\{0,1\}",
    ),
    Equation(
        "selective-coverage.svg",
        r"\operatorname{coverage}(\phi)"
        r"=\mathbb{E}_{\mathrm{tgt}}\!\left[\phi(X)\right]",
    ),
    Equation(
        "weighted-residual-gap.svg",
        r"g_c=\operatorname{avg}_{w}\!\left(y_i-p_i\mid i\in c\right)",
    ),
    Equation(
        "simultaneous-radius.svg",
        r"r_c=\sqrt{\frac{\log(2K/\alpha)}{2\,n_{\mathrm{eff},c}}}",
    ),
    Equation(
        "certified-excess.svg",
        r"\operatorname{excess}_c"
        r"=\max\!\left("
        r"|g_c|-r_c-\rho_c-\gamma-\tau,\ 0"
        r"\right)",
    ),
)


def _latex_document(equation: Equation) -> str:
    return rf"""
\documentclass[preview,border=2pt]{{standalone}}
\usepackage{{amsmath,amssymb}}
\begin{{document}}
$\displaystyle {equation.tex}$
\end{{document}}
""".strip()


def _run(command: list[str], cwd: Path) -> None:
    # The commands are fixed TeX tooling invocations assembled by this script.
    completed = subprocess.run(  # noqa: S603
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: "
            f"{' '.join(command)}\n{completed.stdout}"
        )


def render_equation(equation: Equation) -> None:
    with tempfile.TemporaryDirectory(prefix="shiftstat-equation-") as tmp:
        workdir = Path(tmp)
        tex_path = workdir / "equation.tex"
        tex_path.write_text(_latex_document(equation), encoding="utf-8")

        _run(
            [
                "latex",
                "-halt-on-error",
                "-interaction=batchmode",
                tex_path.name,
            ],
            workdir,
        )
        _run(
            [
                "dvisvgm",
                "--no-fonts",
                "--exact",
                "--bbox=min",
                "--precision=3",
                "-o",
                str(OUTPUT_DIR / equation.filename),
                "equation.dvi",
            ],
            workdir,
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for equation in EQUATIONS:
        render_equation(equation)
        print(f"Rendered {equation.filename}")


if __name__ == "__main__":
    main()
