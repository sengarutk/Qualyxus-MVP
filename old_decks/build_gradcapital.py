"""GradCapital attachment: the 10-minute deck without dhaRti's funding slides, plus the demo link.

    python -I old_decks/build_gradcapital.py <folder holding python-pptx>
"""
import sys

sys.path.insert(0, sys.argv[1])
from pptx import Presentation  # noqa: E402

SRC = "/home/sengar/dev/QUALYXUS_dhaRti_Final_10min.pptx"
OUT = "/home/sengar/dev/QUALYXUS_GradCapital.pptx"
DEMO = "https://drive.google.com/file/d/1ktDP_Sudm2bHiT3ViYH8cp2E7b9oSUZB/view?usp=sharing"
DROP = {10, 11}   # "How dhaRti Can Help" and the dhaRti seed request

prs = Presentation(SRC)
s = {i + 1: slide for i, slide in enumerate(prs.slides)}


def runs_with(slide, text):
    found = [r for sh in slide.shapes if sh.has_text_frame for p in sh.text_frame.paragraphs for r in p.runs
             if text in r.text]
    if len(found) != 1:
        sys.exit(f"expected 1 run with {text!r}, got {len(found)}")
    return found[0]


run = runs_with(s[9], "Company incorporation planned with dhaRti support.")
run.text = run.text.replace("Company incorporation planned with dhaRti support.", "Company incorporation planned.")
label = runs_with(s[12], "Live demo and code walkthrough available on request")
label.text = "Watch the 2:41 software demo"
label.hyperlink.address = DEMO

ids = prs.slides._sldIdLst
for number, entry in reversed(list(enumerate(list(ids), 1))):
    if number in DROP:
        prs.part.drop_rel(entry.rId)
        ids.remove(entry)
prs.save(OUT)
print("saved", len(prs.slides), "slides")
