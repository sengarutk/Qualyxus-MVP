"""Build the 10-minute dhaRti deck from the corrected full deck, which stays unchanged.

    python -I old_decks/build_10min_dhaRti.py <folder holding python-pptx>
"""
import sys

sys.path.insert(0, sys.argv[1])
from pptx import Presentation  # noqa: E402

FULL = "/home/sengar/dev/QUALYXUS_Pitch_Deck_dhaRti.pptx"
OUT = "/home/sengar/dev/QUALYXUS_dhaRti_Final_10min.pptx"
# Problem, solution, proof, traction, why we win, market, money, team, then the ask.
ORDER = [1, 3, 4, 7, 13, 10, 9, 11, 15, 16, 17, 18]

prs = Presentation(FULL)
s = {i + 1: slide for i, slide in enumerate(prs.slides)}


def shape_with(slide, text):
    return next(sh for sh in slide.shapes if sh.has_text_frame and text in sh.text_frame.text)


def set_runs(para, *texts):
    if len(para.runs) != len(texts):
        sys.exit(f"run count differs: {para.text[:60]!r}")
    for run, text in zip(para.runs, texts):
        run.text = text


def drop(shape, starts_with):
    para = next(p for p in shape.text_frame.paragraphs if p.text.startswith(starts_with))
    para._p.getparent().remove(para._p)


# The at-a-glance slide with the stage line is gone, so slide 7 must state the stage.
set_runs(shape_with(s[7], "What the Platform Already Does").text_frame.paragraphs[0],
         "Working Prototype, Pre-Revenue: What the Platform Already Does")

# Business slide: readable in a minute, and it now carries the go-to-market in one line.
set_runs(shape_with(s[11], "Two revenue streams").text_frame.paragraphs[0],
         "How we land: ",
         "60-day paid pilot beside manual inspectors (₹1L, credited on conversion), then 5–8 lines per plant.")
drop(shape_with(s[11], "Usually within plant-level"), "Usually within plant-level")
licence = shape_with(s[11], "Volume: ₹2.0L")
drop(licence, "Volume: ₹2.0L")
set_runs(next(p for p in licence.text_frame.paragraphs if p.text.startswith("Machine-health")),
         "Target software gross margin: ", "80%+")

# Traction: slide 13's layout with the real conversations. Its modelled revenue is removed, so
# assumed figures can never read as numbers for a named company.
by_id = {sh.shape_id: sh for sh in s[13].shapes}
set_runs(by_id[2].text_frame.paragraphs[0], "Traction")
set_runs(by_id[3].text_frame.paragraphs[0], "Early Industry Conversations")
set_runs(by_id[4].text_frame.paragraphs[0],
         "I have spoken with the heads at three manufacturers about piloting Qualyxus on their parts.")
PILOT = ("Pilot: ", "spoke with the heads about testing Qualyxus on their parts.")
CARDS = (
    (7, 8, "Rotary Connectors", [PILOT, ("360° inspector: ", "agreed to help build the design, starting as soon "
                                                             "as funding arrives.")]),
    (11, 12, "Rossell Techsys", [PILOT]),
    (15, 16, "Moog", [PILOT]),
)
for title_id, body_id, name, lines in CARDS:
    set_runs(by_id[title_id].text_frame.paragraphs[0], name)
    paras = list(by_id[body_id].text_frame.paragraphs)
    for para, line in zip(paras, lines):
        set_runs(para, *line)
    for para in paras[len(lines):]:
        para._p.getparent().remove(para._p)
set_runs(by_id[18].text_frame.paragraphs[0],
         "Once funded: ",
         "the 360° inspector design starts with Rotary Connectors and the test rig gets built for trials on "
         "real parts. ",
         "Next Milestone: ",
         "turn these conversations into 60-day paid pilots and publish measured results.")
set_runs(by_id[19].text_frame.paragraphs[0], "Conversations and design support so far. No signed pilots or revenue yet.")

ids = prs.slides._sldIdLst
entries = list(ids)
for number, entry in enumerate(entries, 1):
    if number not in ORDER:
        prs.part.drop_rel(entry.rId)
    ids.remove(entry)
for number in ORDER:
    ids.append(entries[number - 1])
prs.save(OUT)
print("saved", len(prs.slides), "slides")
