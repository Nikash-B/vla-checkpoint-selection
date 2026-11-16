"""Format numbers and build the Markdown tables of the result pages.

Every number on a page comes from the results or the config. Success rates print in percent,
regrets and drops in points, critic values in reward units, and steps as thousands (100k). Each
kind of quantity prints at one precision on every page; the constants below set them.
"""

from ckpt_selection import layout

DASH = "–"
# Decimals of each kind of quantity. Two decimals print the median of two regrets exactly.
REGRET = 2  # regrets, R_last and differences of regrets, in points
SUCCESS_CHANGE = 1  # drops, falls and other changes of success rates, in points
PERCENT = 1  # success rates, in percent
SUCCESS_SE = 1  # standard errors of success rates, in points
MARGIN = 2  # margins between practice-success scores, in points
RHO = 3  # correlations
Q_VALUE = 3  # critic values (held-out Q, replay-batch Q, mean Q, ensemble spread), reward units
SOFTOPC_SCORE = 3  # SoftOPC scores, reward units
ORACLE_BIAS = 3  # bias of the best confirmed success, in points
SCAN_BIAS = 2  # scan success minus confirmed success and its standard error, in points
RELATIVE_L2 = 1  # L2 relative to the base policy, in percent


def number(value, decimals: int) -> str:
    """Return a number at a fixed precision, or a dash for None."""
    if value is None:
        return DASH
    return f"{value:.{decimals}f}"


def percent(rate) -> str:
    """Return a success rate (a fraction) in percent, or a dash for None."""
    if rate is None:
        return DASH
    return number(100 * rate, PERCENT)


def config_percent(rate: float) -> str:
    """Return a rate from the config in percent, as written there (0.2 -> '20')."""
    return f"{100 * rate:g}"


def step(value) -> str:
    """Return a gradient step in thousands, or a dash for None."""
    if value is None:
        return DASH
    return layout.format_step(value)


def steps(values) -> str:
    """Return a list of steps in thousands, comma separated, or a dash for an empty list."""
    return ", ".join(step(value) for value in values) or DASH


def yes_no(flag: bool) -> str:
    """Return a flag as yes or no."""
    return "yes" if flag else "no"


def interval(bounds) -> str:
    """Return a confidence interval as [low, high], or a dash for None."""
    if bounds is None:
        return DASH
    return f"[{number(bounds[0], RHO)}, {number(bounds[1], RHO)}]"


def tags(values) -> str:
    """Return run tags comma separated, or a dash for an empty list."""
    return ", ".join(values) or DASH


def table(headers: list[str], rows: list[list[str]]) -> str:
    """Return a GitHub Markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def bullets(items: list[str]) -> str:
    """Return a Markdown bullet list."""
    return "\n".join(f"- {item}" for item in items)


def table_label(text: str) -> str:
    """Return a bold line that names the table below it."""
    return f"**{text}**"


def all_steps(series: list[dict]) -> list[int]:
    """Return every step that some series has, in step order."""
    return sorted(set().union(*series))


def step_table(first_header: str, rows: list[tuple[str, dict]], cell) -> str:
    """Return a table with one row per (name, {step: value}) and one column per step.

    cell formats one value; a step that a row lacks gets a dash.
    """
    columns = all_steps([values for _, values in rows])
    lines = [
        [name, *(cell(values[column]) if column in values else DASH for column in columns)]
        for name, values in rows
    ]
    return table([first_header, *(step(column) for column in columns)], lines)


def page(title: str, intro: str, sections: list[tuple[str, str, list[str]]]) -> str:
    """Return a page: a heading, an intro, then sections of (heading, intro, blocks)."""
    parts = [f"# {title}", intro]
    for heading, section_intro, blocks in sections:
        parts.append(f"## {heading}")
        if section_intro:
            parts.append(section_intro)
        parts.extend(blocks)
    return "\n\n".join(parts) + "\n"
