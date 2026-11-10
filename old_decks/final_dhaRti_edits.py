"""Final pass on the dhaRti deck. Every replacement must match exactly, or nothing is saved.

    python -I old_decks/final_dhaRti_edits.py <folder holding python-pptx>
"""
import sys

sys.path.insert(0, sys.argv[1])
from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402

DECK = "/home/sengar/dev/QUALYXUS_Pitch_Deck_dhaRti.pptx"
prs = Presentation(DECK)
s = list(prs.slides)


def paragraphs(slide):
    for shape in slide.shapes:
        if shape.has_text_frame:
            yield from shape.text_frame.paragraphs
        if getattr(shape, "has_table", False) and shape.has_table:
            for row in shape.table.rows:
                for cell in row.cells:
                    yield from cell.text_frame.paragraphs


def swap(slide, old, new):
    hits = 0
    for para in paragraphs(slide):
        for run in para.runs:
            if old in run.text:
                run.text = run.text.replace(old, new)
                hits += 1
    if hits != 1:
        sys.exit(f"expected 1 match, got {hits}: {old[:70]!r}")


# Slide 1: early warning is not proven, so do not promise to stop scrap before it happens.
swap(s[0], "catch", "trace")
swap(s[0], "process drift and machine wear before they become scrap or downtime, with all data",
     "defects to process drift and machine wear, so causes get fixed sooner, with all data")
swap(s[1], "111 automated tests", "166 automated tests")
swap(s[5], "A bad image is a camera fault, not a reject.",
     "A bad image or a part no camera saw is flagged not inspected, never passed.")

# Slide 7: latest measurements, the catch rate behind the AUROC, and what the tests now cover.
swap(s[6], "(previous baseline: 0.56–0.96).",
     "(previous baseline: 0.56–0.96). With thresholds set for low false rejects, it caught 35–90% "
     "of defects, depending on the part.")
swap(s[6], "2 of 200 unseen good parts", "2 of 400 unseen good parts")
swap(s[6], "111", "166")
swap(s[6], "Automated tests passing, plus end-to-end offline runs with every part judged correctly and "
           "the audit log intact (simulation).",
     "Automated tests passing, including forced failures: dead cameras, PLC outage, power cuts. "
     "End-to-end offline runs keep the audit log intact (simulation).")
swap(s[6], "Nothing has run on a real PLC, Jetson or factory line yet",
     "Nothing has run on a real PLC, camera, Jetson or factory line yet")

swap(s[8], "Qualyxus estimate (see notes).", "Qualyxus estimate.")

# Slide 13: no plant has agreed to a pilot, so say so and do not imply these lines are lined up.
swap(s[12], "Projected Impact on Three Target Pilot Lines",
     "Pre-Revenue Today: Projected Impact on Three Target Pilot Lines")
swap(s[12], "convert these lines to 60-day paid pilots and publish",
     "sign lines like these for 60-day paid pilots and publish")
band = next(sh for sh in s[12].shapes if sh.has_text_frame is True and sh.shape_type == 1
            and sh.fill.type == 1 and str(sh.fill.fore_color.rgb) == "B8F2B0")
band.fill.solid()
band.fill.fore_color.rgb = RGBColor(0xE6, 0xE7, 0xF1)   # the deck's own band colour

# Slide 14 must agree with slide 17: the first paid conversion is month 9.
swap(s[13], "Projections are illustrative planning assumptions, not forecasts.",
     "Year 1 starts at the first paid conversion (month 9 of the seed plan). "
     "Projections are illustrative planning assumptions, not forecasts.")

prs.save(DECK)
print("saved")
