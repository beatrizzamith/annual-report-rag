"""Builds a small fixture PDF for integration tests: a two-column narrative
page (with a known FTE sentence and a known sustainability goal sentence)
plus a table page with real, PyMuPDF-detectable table borders.

Generated at test time rather than committed as a binary, so the fixture's
content stays reviewable as ordinary Python.
"""

from pathlib import Path

import pymupdf

FTE_SENTENCE = (
    "The Group employed an average of 12,345 FTE during the year, "
    "measured as full-time equivalents across all operating segments."
)
GOAL_SENTENCE = (
    "We aim to reach net-zero Scope 1 and 2 emissions by 2035, compared with a 2019 baseline."
)
LEFT_COLUMN = (
    "Our people are central to everything we do. " * 3
    + FTE_SENTENCE
    + " Employee wellbeing programmes were expanded across every region."
)
RIGHT_COLUMN = (
    "Sustainability remains a strategic priority for the Group. "
    + GOAL_SENTENCE
    + " Progress against this target is reviewed by the board each quarter."
)

TABLE_HEADER = ["Region", "FTE 2025", "FTE 2024"]
TABLE_ROWS = [
    ["EMEA", "6,145", "5,900"],
    ["Americas", "3,200", "3,050"],
    ["Asia Pacific", "3,000", "2,850"],
]


def build_fixture_pdf(path: Path) -> Path:
    doc = pymupdf.open()

    page1 = doc.new_page(width=595, height=842)  # A4
    page1.insert_textbox(pymupdf.Rect(72, 72, 280, 700), LEFT_COLUMN, fontsize=10)
    page1.insert_textbox(pymupdf.Rect(310, 72, 520, 700), RIGHT_COLUMN, fontsize=10)

    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((72, 72), "Workforce by region", fontsize=14)
    col_widths = [150, 100, 100]
    row_height = 20
    top = 110
    for row_index, row in enumerate([TABLE_HEADER, *TABLE_ROWS]):
        y = top + row_index * row_height
        x = 72
        for col_index, cell in enumerate(row):
            width = col_widths[col_index]
            page2.draw_rect(pymupdf.Rect(x, y, x + width, y + row_height))
            page2.insert_text((x + 4, y + row_height - 6), cell, fontsize=10)
            x += width

    doc.save(path)
    doc.close()
    return path
