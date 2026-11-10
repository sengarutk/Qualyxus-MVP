"""Keep named companies away from the modelled revenue on the full deck's slide 13.

    python -I old_decks/traction_fix_full_deck.py <folder holding python-pptx>
"""
import sys

sys.path.insert(0, sys.argv[1])
from pptx import Presentation  # noqa: E402

DECK = "/home/sengar/dev/QUALYXUS_Pitch_Deck_dhaRti.pptx"
prs = Presentation(DECK)
s = list(prs.slides)


def swap(slide, old, new):
    hits = 0
    for shape in slide.shapes:
        if shape.has_text_frame:
            for para in shape.text_frame.paragraphs:
                for run in para.runs:
                    if old in run.text:
                        run.text = run.text.replace(old, new)
                        hits += 1
    if hits != 1:
        sys.exit(f"expected 1 match, got {hits}: {old[:70]!r}")


swap(s[12], "turn talks with Rossell Techsys, Rotary Connectors and Moog into 60-day paid pilots",
     "sign lines like these for 60-day paid pilots")
swap(s[12], "Projections, not measured results. Assumed line output is illustrative;",
     "Projections, not measured results, for example lines, not any named company. "
     "Assumed line output is illustrative;")
swap(s[1], "results so far are from public benchmarks and simulation.",
     "results so far are from public benchmarks and simulation. Spoke with the heads at Rossell Techsys, "
     "Rotary Connectors and Moog about pilots.")
prs.save(DECK)
print("saved")
