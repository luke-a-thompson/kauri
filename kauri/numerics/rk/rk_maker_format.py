from typing import TYPE_CHECKING

import sympy

if TYPE_CHECKING:
    from kauri.numerics.rk.rk_maker import SolveResult


def _result_metadata(result: "SolveResult") -> list[tuple[str, str]]:
    fixings_str = (
        ", ".join(f"{k}={sympy.sstr(v)}" for k, v in result.fixings.items())
        if result.fixings
        else "none"
    )
    return [
        ("ansatz", result.ansatz),
        ("reduced equations", str(len(result.equations))),
        ("active unknowns", ", ".join(result.unknowns)),
        ("free symbols", ", ".join(result.free_symbols)),
        ("fixings", fixings_str),
    ]


def result_to_text(
    result: "SolveResult",
    max_cell_chars: int = 48,
) -> str:
    lines: list[str] = []
    lines.append("=== Explicit RK Maker Result ===")
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
        lines.append("")
        lines.append(
            "Use build_method_from_ansatz return value `methods` for constructed methods."
            " This object only stores solve metadata and symbolic solutions."
        )
    return "\n".join(lines)


def result_to_latex(
    result: "SolveResult",
    max_cell_chars: int = 48,
    standalone: bool = True,
) -> str:
    lines: list[str] = []
    if standalone:
        lines.append(r"\documentclass{article}")
        lines.append(r"\usepackage{amsmath}")
        lines.append(r"\begin{document}")

    lines.append(r"\section*{Explicit RK Maker Result}")
    lines.append(r"\begin{itemize}")
    for key, value in _result_metadata(result):
        lines.append(rf"\item {key}: \texttt{{{value}}}")
    lines.append(r"\end{itemize}")

    if len(result.solutions) > 0:
        lines.append(
            r"\textit{Use make\_explicit\_rk\_methods return value methods for constructed methods."
            r" This object only stores solve metadata and symbolic solutions.}"
        )

    if standalone:
        lines.append(r"\end{document}")
    return "\n".join(lines)


def format_tableau_text(
    c_vector: list[sympy.core.basic.Basic],
    a_matrix: list[list[sympy.core.basic.Basic]],
    b_vector: list[sympy.core.basic.Basic],
    max_cell_chars: int = 48,
) -> str:
    tableau_str, definitions = _format_text_tableau_structure(
        c_vector=c_vector,
        a_matrix=a_matrix,
        b_vector=b_vector,
        max_cell_chars=max_cell_chars,
    )
    if len(definitions) == 0:
        return tableau_str
    return "\n".join([tableau_str, "", "definitions:", *[f"  {d}" for d in definitions]])


def format_tableau_latex(
    c_vector: list[sympy.core.basic.Basic],
    a_matrix: list[list[sympy.core.basic.Basic]],
    b_vector: list[sympy.core.basic.Basic],
    max_cell_chars: int = 48,
) -> str:
    tableau_latex, definitions = _format_latex_tableau_structure(
        c_vector=c_vector,
        a_matrix=a_matrix,
        b_vector=b_vector,
        max_cell_chars=max_cell_chars,
    )
    if len(definitions) == 0:
        return "\n".join([r"\[", tableau_latex, r"\]"])
    lines: list[str] = [r"\[", tableau_latex, r"\]", r"\[", r"\begin{aligned}"]
    for name, expr in definitions:
        lines.append(rf"{name} &= {sympy.latex(expr)}\\")
    lines.extend([r"\end{aligned}", r"\]"])
    return "\n".join(lines)


def _format_text_tableau_from_strings(
    c_cells: list[str],
    a_cells: list[list[str]],
    b_cells: list[str],
) -> str:
    stages = len(b_cells)
    c_width = max(len(s) for s in c_cells + ["c"])
    a_widths: list[int] = []
    for j in range(stages):
        col = [a_cells[i][j] for i in range(stages)] + [b_cells[j]]
        a_widths.append(max(len(s) for s in col + ["a"]))

    def pad(s: str, w: int) -> str:
        return s + " " * (w - len(s))

    lines: list[str] = []
    for i in range(stages):
        left = pad(c_cells[i], c_width)
        right = " ".join(pad(a_cells[i][j], a_widths[j]) for j in range(stages))
        lines.append(f"{left} | {right}")

    sep_len = c_width + 3 + sum(a_widths) + max(0, stages - 1)
    lines.append("-" * sep_len)
    b_row = " ".join(pad(b_cells[j], a_widths[j]) for j in range(stages))
    lines.append(f"{' ' * c_width} | {b_row}")
    return "\n".join(lines)


def _format_text_tableau_structure(
    c_vector: list[sympy.core.basic.Basic],
    a_matrix: list[list[sympy.core.basic.Basic]],
    b_vector: list[sympy.core.basic.Basic],
    max_cell_chars: int,
) -> tuple[str, list[str]]:
    def maybe_placeholder(raw: str, expr: sympy.core.basic.Basic) -> str:
        if len(raw) <= max_cell_chars:
            return raw
        key = sympy.sstr(expr)
        if key not in placeholder_by_key:
            name = f"E{len(placeholder_by_key) + 1}"
            placeholder_by_key[key] = name
            definitions.append(f"{name} = {raw}")
        return placeholder_by_key[key]

    stages = len(b_vector)
    placeholder_by_key: dict[str, str] = {}
    definitions: list[str] = []

    c_cells = [maybe_placeholder(sympy.sstr(c_vector[i]), c_vector[i]) for i in range(stages)]

    a_cells: list[list[str]] = []
    for i in range(stages):
        row: list[str] = []
        for j in range(stages):
            raw = sympy.sstr(a_matrix[i][j])
            row.append(maybe_placeholder(raw, a_matrix[i][j]))
        a_cells.append(row)

    b_cells = [maybe_placeholder(sympy.sstr(b_vector[j]), b_vector[j]) for j in range(stages)]
    return _format_text_tableau_from_strings(c_cells, a_cells, b_cells), definitions


def _format_latex_tableau_structure(
    c_vector: list[sympy.core.basic.Basic],
    a_matrix: list[list[sympy.core.basic.Basic]],
    b_vector: list[sympy.core.basic.Basic],
    max_cell_chars: int,
) -> tuple[str, list[tuple[str, sympy.core.basic.Basic]]]:
    def cell(expr: sympy.core.basic.Basic) -> str:
        raw = sympy.sstr(expr)
        if len(raw) <= max_cell_chars:
            return sympy.latex(expr)
        key = raw
        if key not in placeholder_by_key:
            idx = len(placeholder_by_key) + 1
            placeholder_by_key[key] = idx
            definitions.append((rf"E_{{{idx}}}", expr))
        return rf"E_{{{placeholder_by_key[key]}}}"

    stages = len(b_vector)
    placeholder_by_key: dict[str, int] = {}
    definitions: list[tuple[str, sympy.core.basic.Basic]] = []

    cols = "c|" + ("c" * stages)
    lines: list[str] = [rf"\begin{{array}}{{{cols}}}"]
    for i in range(stages):
        row = [cell(c_vector[i])] + [cell(a_matrix[i][j]) for j in range(stages)]
        lines.append(" & ".join(row) + r"\\")
    lines.append(r"\hline")
    brow = [""] + [cell(b_vector[j]) for j in range(stages)]
    lines.append(" & ".join(brow) + r"\\")
    lines.append(r"\end{array}")
    return "\n".join(lines), definitions
