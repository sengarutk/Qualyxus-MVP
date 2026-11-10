"""Add the company conversations to the full deck and strip filler wording.

Facts come from the founder (2026-11-07). "Spoke with" stays "spoke with": only Rotary Connectors'
design help is stated as agreed. Every replacement must match exactly, or nothing is saved.

    python -I old_decks/traction_edits.py <folder holding python-pptx>
"""
import sys

sys.path.insert(0, sys.argv[1])
from pptx import Presentation  # noqa: E402

DECK = "/home/sengar/dev/QUALYXUS_Pitch_Deck_dhaRti.pptx"
prs = Presentation(DECK)
s = list(prs.slides)


def paragraphs(slide):
    for shape in slide.shapes:
        if shape.has_text_frame:
            yield from shape.text_frame.paragraphs


def swap(slide, old, new):
    hits = 0
    for para in paragraphs(slide):
        for run in para.runs:
            if old in run.text:
                run.text = run.text.replace(old, new)
                hits += 1
    if hits != 1:
        sys.exit(f"expected 1 match, got {hits}: {old[:70]!r}")


for label in ("Startup Name:", "Prepared for:"):
    para = next(p for p in paragraphs(s[0]) if p.text.startswith(label))
    para._p.getparent().remove(para._p)

swap(s[2], "Compelling Problem Statement", "The Problem")
swap(s[2], "The Problem: Quality and Machine Data", "Quality and Machine Data")
swap(s[3], "Solution Offered & Innovation Proposed", "The Solution")
swap(s[3], "The Causal Bridge: Every Defect", "Every Defect")
swap(s[8], "Market Research & Opportunity", "Market Opportunity")
swap(s[9], "Competition & Comparison Matrix", "Competition")
swap(s[10], "Business Model & Unit Economics", "Business Model")
swap(s[14], "Team & Advisory Details", "Team & Advisors")

swap(s[7], '"All-Faces" 360° multi-camera enclosure',
     '"All-Faces" 360° enclosure (design with Rotary Connectors starts on funding)')
swap(s[12], "sign lines like these for 60-day paid pilots",
     "turn talks with Rossell Techsys, Rotary Connectors and Moog into 60-day paid pilots")
swap(s[16], "Test rig, first station, company setup and patent filing.",
     "Test rig, first station, 360° inspector design start, company setup and patent filing.")
prs.save(DECK)
print("full deck saved")
