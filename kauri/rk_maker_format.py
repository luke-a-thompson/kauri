from typing import Literal, TYPE_CHECKING

import sympy

if TYPE_CHECKING:
    from kauri.rk_maker import RKMakerResult


def result_to_text(
    result: "RKMakerResult",
    mode: Literal["structure", "symbolic"] = "structure",
    max_cell_chars: int = 48,
) -> str:
    lines: list[str] = []
    lines.append("=== Explicit RK Maker Result ===")
    lines.append(f"solver: {result.solver}")
    lines.append(f"ansatz: {result.ansatz}")
    lines.append(f"equations: {len(result.equations)}")
    lines.append(f"active unknowns: {result.unknowns}")
    lines.append(f"free symbols: {result.free_symbols}")
    if len(result.free_symbols) > 0:
        if len(result.free_symbol_relations) == 0:
            lines.append("relations among free symbols: none found")
        else:
            lines.append("relations among free symbols:")
            for relation in result.free_symbol_relations:
                lines.append(f"  {sympy.sstr(relation)} = 0")
    lines.append(f"method count: {len(result.methods)}")

    if len(result.solutions) == 0:
        return "\n".join(lines)

    c_vector, a_matrix, b_vector = _tableau_exprs_from_solution(result.solutions[0])
    lines.append("")
    if mode == "symbolic":
        lines.append(_format_text_tableau(c_vector, a_matrix, b_vector))
        return "\n".join(lines)

    tableau_str, definitions = _format_text_tableau_structure(
        c_vector=c_vector,
        a_matrix=a_matrix,
        b_vector=b_vector,
        max_cell_chars=max_cell_chars,
    )
    lines.append(tableau_str)
    if len(definitions) > 0:
        lines.append("")
        lines.append("definitions:")
        lines.extend(f"  {d}" for d in definitions)
    return "\n".join(lines)


def result_to_latex(
    result: "RKMakerResult",
    mode: Literal["structure", "symbolic"] = "structure",
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
    lines.append(rf"\item solver: \texttt{{{result.solver}}}")
    lines.append(rf"\item ansatz: \texttt{{{result.ansatz}}}")
    lines.append(rf"\item equations: {len(result.equations)}")
    lines.append(rf"\item active unknowns: \texttt{{{', '.join(result.unknowns)}}}")
    lines.append(rf"\item free symbols: \texttt{{{', '.join(result.free_symbols)}}}")
    lines.append(rf"\item method count: {len(result.methods)}")
    lines.append(r"\end{itemize}")

    if len(result.solutions) > 0:
        c_vector, a_matrix, b_vector = _tableau_exprs_from_solution(result.solutions[0])
        if mode == "symbolic":
            tableau_latex = _format_latex_tableau(c_vector, a_matrix, b_vector)
            lines.append(r"\[")
            lines.append(tableau_latex)
            lines.append(r"\]")
        else:
            tableau_latex, definitions = _format_latex_tableau_structure(
                c_vector=c_vector,
                a_matrix=a_matrix,
                b_vector=b_vector,
                max_cell_chars=max_cell_chars,
            )
            lines.append(r"\[")
            lines.append(tableau_latex)
            lines.append(r"\]")
            if len(definitions) > 0:
                lines.append(r"\[")
                lines.append(r"\begin{aligned}")
                for name, expr in definitions:
                    lines.append(rf"{name} &= {sympy.latex(expr)}\\")
                lines.append(r"\end{aligned}")
                lines.append(r"\]")

    if standalone:
        lines.append(r"\end{document}")
    return "\n".join(lines)


def _tableau_exprs_from_solution(
    named_solution: dict[str, sympy.core.basic.Basic],
) -> tuple[
    list[sympy.core.basic.Basic],
    list[list[sympy.core.basic.Basic]],
    list[sympy.core.basic.Basic],
]:
    stages = _infer_stages_from_named_solution(named_solution)
    a_matrix: list[list[sympy.core.basic.Basic]] = [
        [sympy.Integer(0) for _ in range(stages)] for _ in range(stages)
    ]
    b_vector: list[sympy.core.basic.Basic] = [sympy.Integer(0) for _ in range(stages)]
    for i in range(stages):
        for j in range(i):
            a_matrix[i][j] = sympy.sympify(named_solution.get(f"a{i}{j}", sympy.Integer(0)))
        b_vector[i] = sympy.sympify(named_solution.get(f"b{i}", sympy.Integer(0)))

    c_vector: list[sympy.core.basic.Basic] = []
    for i in range(stages):
        row_sum = sympy.Integer(0)
        for j in range(stages):
            row_sum = sympy.simplify(row_sum + a_matrix[i][j])
        c_vector.append(sympy.simplify(row_sum))
    return c_vector, a_matrix, b_vector


def _infer_stages_from_named_solution(named_solution: dict[str, sympy.core.basic.Basic]) -> int:
    stages = 0
    for key in named_solution.keys():
        if key.startswith("b"):
            try:
                stages = max(stages, int(key[1:]) + 1)
            except ValueError:
                continue
    return stages


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


def _format_text_tableau(
    c_vector: list[sympy.core.basic.Basic],
    a_matrix: list[list[sympy.core.basic.Basic]],
    b_vector: list[sympy.core.basic.Basic],
) -> str:
    stages = len(b_vector)
    c_cells = [sympy.sstr(expr) for expr in c_vector]
    a_cells = [[sympy.sstr(a_matrix[i][j]) for j in range(stages)] for i in range(stages)]
    b_cells = [sympy.sstr(expr) for expr in b_vector]
    return _format_text_tableau_from_strings(c_cells, a_cells, b_cells)


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


def _format_latex_tableau(
    c_vector: list[sympy.core.basic.Basic],
    a_matrix: list[list[sympy.core.basic.Basic]],
    b_vector: list[sympy.core.basic.Basic],
) -> str:
    stages = len(b_vector)
    cols = "c|" + ("c" * stages)
    lines: list[str] = [rf"\begin{{array}}{{{cols}}}"]
    for i in range(stages):
        row = [sympy.latex(c_vector[i])] + [sympy.latex(a_matrix[i][j]) for j in range(stages)]
        lines.append(" & ".join(row) + r"\\")
    lines.append(r"\hline")
    brow = [""] + [sympy.latex(b_vector[j]) for j in range(stages)]
    lines.append(" & ".join(brow) + r"\\")
    lines.append(r"\end{array}")
    return "\n".join(lines)


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
