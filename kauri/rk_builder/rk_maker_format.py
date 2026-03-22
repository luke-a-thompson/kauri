from typing import Literal

import sympy

from kauri.methods.rk import ButcherTableau
from kauri.methods.williamson import WilliamsonRecursion
from kauri.rk_builder._rk_maker_core import SolveResult
from kauri.rk_builder.williamson import williamson_tableau_expressions

WilliamsonRenderMode = Literal["add", "exp"]


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
        ("objective-evaluated methods", str(len(result.objective_scores))),
    ]


def _objective_scores_text(result: SolveResult) -> list[str]:
    if len(result.objective_scores) == 0:
        return []
    lines = ["objective scores:"]
    for method_score in result.objective_scores:
        score_text = ", ".join(
            f"{objective}={value:.12g}" for objective, value in method_score.scores.items()
        )
        lines.append(f"  {method_score.method_name}: {score_text}")
    return lines


def _family_summary_text(result: SolveResult) -> str | None:
    if len(result.solutions) == 0 or len(result.free_symbols) == 0:
        return None
    free_symbols = ", ".join(result.free_symbols)
    return (
        f"No concrete method found; the solver returned a family parameterised by [{free_symbols}]."
    )


def result_to_text(result: SolveResult) -> str:
    lines: list[str] = ["=== Explicit RK Maker Result ==="]
    for key, value in _result_metadata(result):
        lines.append(f"{key}: {value}")
    family_summary = _family_summary_text(result)
    if family_summary is not None:
        lines.append(family_summary)
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
    lines.extend(_objective_scores_text(result))
    return "\n".join(lines)


def result_to_latex(result: SolveResult, standalone: bool = True) -> str:
    lines: list[str] = []
    if standalone:
        lines += [r"\documentclass{article}", r"\usepackage{amsmath}", r"\begin{document}"]
    lines += [r"\section*{Explicit RK Maker Result}", r"\begin{itemize}"]
    for key, value in _result_metadata(result):
        lines.append(rf"\item {key}: \texttt{{{value}}}")
    lines.append(r"\end{itemize}")
    family_summary = _family_summary_text(result)
    if family_summary is not None:
        lines.append(rf"\textit{{{family_summary}}}")
    if len(result.solutions) > 0:
        lines.append(
            r"\textit{Use the builder return value methods for constructed methods."
            r" This object only stores solve metadata and symbolic solutions.}"
        )
    if len(result.objective_scores) > 0:
        lines.append(r"\textbf{Objective scores:}")
        lines.append(r"\begin{itemize}")
        for method_score in result.objective_scores:
            score_text = ", ".join(
                rf"{objective}={value:.12g}" for objective, value in method_score.scores.items()
            )
            lines.append(rf"\item \texttt{{{method_score.method_name}}}: {score_text}")
        lines.append(r"\end{itemize}")
    if standalone:
        lines.append(r"\end{document}")
    return "\n".join(lines)


def format_tableau_text(
    tableau: ButcherTableau, max_cell_chars: int = 48, *, rationalize: bool = True
) -> str:
    s = tableau.stages
    placeholder_by_key: dict[str, str] = {}
    definitions: list[str] = []

    def to_expr(val: object) -> object:
        if rationalize:
            return sympy.nsimplify(val, tolerance=1e-3, rational=True)
        return val

    def cell(expr: object) -> str:
        raw = sympy.sstr(to_expr(expr))
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


def format_tableau_latex(
    tableau: ButcherTableau, max_cell_chars: int = 48, *, rationalize: bool = False
) -> str:
    s = tableau.stages
    placeholder_by_key: dict[str, int] = {}
    definitions: list[tuple[str, object]] = []

    def to_expr(val: object) -> object:
        if rationalize:
            return sympy.nsimplify(val, tolerance=1e-3, rational=True)
        return val

    def cell(expr: object) -> str:
        converted = to_expr(expr)
        raw = sympy.sstr(converted)
        if len(raw) <= max_cell_chars:
            return sympy.latex(converted)
        if raw not in placeholder_by_key:
            idx = len(placeholder_by_key) + 1
            placeholder_by_key[raw] = idx
            definitions.append((rf"E_{{{idx}}}", converted))
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
        array_lines.append(
            r"\hat{b} & " + " & ".join(cell(tableau.b_hat[j]) for j in range(s)) + r"\\"
        )
    array_lines.append(r"\end{array}")
    tableau_latex = "\n".join(array_lines)

    if not definitions:
        return "\n".join([r"\[", tableau_latex, r"\]"])
    lines = [r"\[", tableau_latex, r"\]", r"\[", r"\begin{aligned}"]
    for name, expr in definitions:
        lines.append(rf"{name} &= {sympy.latex(expr)}\\")
    lines += [r"\end{aligned}", r"\]"]
    return "\n".join(lines)


def _expanded_williamson_data(
    recursion: WilliamsonRecursion,
) -> tuple[list[sympy.Expr], list[sympy.Expr], list[sympy.Expr]]:
    storage_a = [
        sympy.simplify(recursion.A[i_idx][i_idx - 1] if i_idx > 0 else sympy.Integer(0))
        for i_idx in range(recursion.stages)
    ]
    a_expr, _ = williamson_tableau_expressions(
        stages=recursion.stages,
        A_symbols=storage_a,
        B_symbols=recursion.B,
    )
    stage_nodes = [
        sympy.simplify(sum(a_expr[i_idx][j_idx] for j_idx in range(recursion.stages)))
        for i_idx in range(recursion.stages)
    ]
    step_b = [sympy.simplify(value) for value in recursion.B]
    return stage_nodes, storage_a, step_b


def format_williamson_recursion_text(
    recursion: WilliamsonRecursion,
    name: str | None = None,
    *,
    title: str = "=== Williamson 2N Method ===",
    mode: WilliamsonRenderMode = "add",
    exponentials_per_timestep: int | None = None,
) -> str:
    stage_nodes, storage_a, step_b = _expanded_williamson_data(recursion)
    lines: list[str] = [title]
    if name is not None:
        lines.append(f"name: {name}")
    lines.append(f"stages: {recursion.stages}")
    if exponentials_per_timestep is not None:
        lines.append(f"exponentials per timestep: {exponentials_per_timestep}")
    lines += ["", "equations:", "  Y_0 = Y_t", "  ΔY_0 = 0"]
    for stage_idx in range(recursion.stages):
        stage_num = stage_idx + 1
        lines.append(
            f"  K_{stage_num} = F(t + ({sympy.sstr(stage_nodes[stage_idx])})*h, Y_{stage_num - 1})"
        )
        lines.append(
            "  "
            f"ΔY_{stage_num} = ({sympy.sstr(storage_a[stage_idx])})"
            f"*ΔY_{stage_num - 1} + h*K_{stage_num}"
        )
        coeff = sympy.sstr(step_b[stage_idx])
        update_line = (
            f"Y_{stage_num} = Y_{stage_num - 1} + ({coeff})*ΔY_{stage_num}"
            if mode == "add"
            else f"Y_{stage_num} = exp(({coeff})*ΔY_{stage_num}) * Y_{stage_num - 1}"
        )
        lines.append("  " + update_line)
    lines.append(f"  Y_(t+h) = Y_{recursion.stages}")
    return "\n".join(lines)


def format_williamson_recursion_latex(
    recursion: WilliamsonRecursion,
    name: str | None = None,
    *,
    heading: str = "Williamson 2N Method",
    mode: WilliamsonRenderMode = "add",
    exponentials_per_timestep: int | None = None,
) -> str:
    stage_nodes, storage_a, step_b = _expanded_williamson_data(recursion)
    lines: list[str] = [r"\["]
    if name is not None:
        lines += [rf"\textbf{{{heading}: }}\texttt{{{name}}}", r"\\"]
    lines.append(r"\begin{aligned}")
    if exponentials_per_timestep is not None:
        lines.append(rf"\text{{exponentials per timestep: }}{exponentials_per_timestep}\\")
    lines += [r"Y_0 &= Y_t\\", r"\Delta Y_0 &= 0\\"]
    for stage_idx in range(recursion.stages):
        stage_num = stage_idx + 1
        lines.append(
            rf"K_{{{stage_num}}} &= F\left(t + ({sympy.latex(stage_nodes[stage_idx])})h, "
            rf"Y_{{{stage_num - 1}}}\right)\\"
        )
        lines.append(
            rf"\Delta Y_{{{stage_num}}} &= ({sympy.latex(storage_a[stage_idx])})"
            rf"\Delta Y_{{{stage_num - 1}}} + hK_{{{stage_num}}}\\"
        )
        coeff = sympy.latex(step_b[stage_idx])
        update_line = (
            rf"Y_{{{stage_num}}} &= Y_{{{stage_num - 1}}} + ({coeff})\Delta Y_{{{stage_num}}}\\"
            if mode == "add"
            else rf"Y_{{{stage_num}}} &= \exp\left(({coeff})\Delta Y_{{{stage_num}}}\right) "
            rf"Y_{{{stage_num - 1}}}\\"
        )
        lines.append(update_line)
    lines += [rf"Y_{{t+h}} &= Y_{{{recursion.stages}}}", r"\end{aligned}", r"\]"]
    return "\n".join(lines)
