from typing import TYPE_CHECKING

import sympy

from kauri.methods.rk import ButcherTableau

from kauri.rk_builder.rk_maker import SolveResult


def _result_metadata(result: SolveResult) -> list[tuple[str, str]]:
    fixings_str = (
        ", ".join(f"{k}={sympy.sstr(v)}" for k, v in result.fixings.items())
        if result.fixings
        else "none"
    )
    return [
        ("parameterization", result.parameterization),
        ("reduced equations", str(len(result.equations))),
        ("active unknowns", ", ".join(result.unknowns)),
        ("free symbols", ", ".join(result.free_symbols)),
        ("fixings", fixings_str),
    ]


def result_to_text(result: SolveResult) -> str:
    lines: list[str] = ["=== Explicit RK Maker Result ==="]
    for key, value in _result_metadata(result):
        lines.append(f"{key}: {value}")
    if len(result.free_symbols) > 0:
        if len(result.free_symbol_relations) == 0:
            lines.append("relations among free symbols: none found")
        else:
            lines.append("relations among free symbols:")
            for relation in result.free_symbol_relations:
                lines.append(f"  {sympy.sstr(relation)} = 0")
    if len(result.solutions) > 0:
        lines.extend(
            [
                "",
                "Use the builder return value `methods` for constructed methods."
                " This object only stores solve metadata and symbolic solutions.",
            ]
        )
    return "\n".join(lines)


def result_to_latex(result: SolveResult, standalone: bool = True) -> str:
    lines: list[str] = []
    if standalone:
        lines += [r"\documentclass{article}", r"\usepackage{amsmath}", r"\begin{document}"]
    lines += [r"\section*{Explicit RK Maker Result}", r"\begin{itemize}"]
    for key, value in _result_metadata(result):
        lines.append(rf"\item {key}: \texttt{{{value}}}")
    lines.append(r"\end{itemize}")
    if len(result.solutions) > 0:
        lines.append(
            r"\textit{Use the builder return value methods for constructed methods."
            r" This object only stores solve metadata and symbolic solutions.}"
        )
    if standalone:
        lines.append(r"\end{document}")
    return "\n".join(lines)


def format_tableau_text(tableau: ButcherTableau, max_cell_chars: int = 48) -> str:
    s = tableau.stages
    placeholder_by_key: dict[str, str] = {}
    definitions: list[str] = []

    def cell(expr: object) -> str:
        raw = sympy.sstr(expr)
        if len(raw) <= max_cell_chars:
            return raw
        if raw not in placeholder_by_key:
            name = f"E{len(placeholder_by_key) + 1}"
            placeholder_by_key[raw] = name
            definitions.append(f"{name} = {raw}")
        return placeholder_by_key[raw]

    c_cells = [cell(tableau.c[i]) for i in range(s)]
    a_cells = [[cell(tableau.a[i][j]) for j in range(s)] for i in range(s)]
    b_cells = [cell(tableau.b[j]) for j in range(s)]
    b_hat_cells = [cell(tableau.b_hat[j]) for j in range(s)] if tableau.b_hat is not None else None

    row_labels = ["c"]
    if b_hat_cells is not None:
        row_labels.append("b_hat")
    c_width = max(len(x) for x in c_cells + row_labels)
    a_widths = [
        max(
            [len(a_cells[i][j]) for i in range(s)]
            + [len(b_cells[j]), len(b_hat_cells[j]) if b_hat_cells is not None else 0, 1]
        )
        for j in range(s)
    ]

    def pad(x: str, w: int) -> str:
        return x + " " * (w - len(x))

    def row(i: int) -> str:
        a_str = " ".join(pad(a_cells[i][j], a_widths[j]) for j in range(s))
        return f"{pad(c_cells[i], c_width)} | {a_str}"

    rows = [row(i) for i in range(s)]
    sep = "-" * (c_width + 3 + sum(a_widths) + max(0, s - 1))
    b_row = f"{' ' * c_width} | {' '.join(pad(b_cells[j], a_widths[j]) for j in range(s))}"
    tableau_lines = [*rows, sep, b_row]
    if b_hat_cells is not None:
        b_hat_row = f"{pad('b_hat', c_width)} | {' '.join(pad(b_hat_cells[j], a_widths[j]) for j in range(s))}"
        tableau_lines.append(b_hat_row)
    tableau_str = "\n".join(tableau_lines)

    if not definitions:
        return tableau_str
    return "\n".join([tableau_str, "", "definitions:", *[f"  {d}" for d in definitions]])


def format_tableau_latex(tableau: ButcherTableau, max_cell_chars: int = 48) -> str:
    s = tableau.stages
    placeholder_by_key: dict[str, int] = {}
    definitions: list[tuple[str, object]] = []

    def cell(expr: object) -> str:
        raw = sympy.sstr(expr)
        if len(raw) <= max_cell_chars:
            return sympy.latex(expr)
        if raw not in placeholder_by_key:
            idx = len(placeholder_by_key) + 1
            placeholder_by_key[raw] = idx
            definitions.append((rf"E_{{{idx}}}", expr))
        return rf"E_{{{placeholder_by_key[raw]}}}"

    cols = "c|" + "c" * s
    array_lines = [rf"\begin{{array}}{{{cols}}}"]
    for i in range(s):
        row = [cell(tableau.c[i])] + [cell(tableau.a[i][j]) for j in range(s)]
        array_lines.append(" & ".join(row) + r"\\")
    array_lines += [
        r"\hline",
        " & ".join([""] + [cell(tableau.b[j]) for j in range(s)]) + r"\\",
    ]
    if tableau.b_hat is not None:
        array_lines.append(r"\hat{b} & " + " & ".join(cell(tableau.b_hat[j]) for j in range(s)) + r"\\")
    array_lines.append(r"\end{array}")
    tableau_latex = "\n".join(array_lines)

    if not definitions:
        return "\n".join([r"\[", tableau_latex, r"\]"])
    lines = [r"\[", tableau_latex, r"\]", r"\[", r"\begin{aligned}"]
    for name, expr in definitions:
        lines.append(rf"{name} &= {sympy.latex(expr)}\\")
    lines += [r"\end{aligned}", r"\]"]
    return "\n".join(lines)
