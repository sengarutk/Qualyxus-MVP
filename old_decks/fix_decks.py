"""Correct the pitch decks in place so every claim matches what the code does today.

Starts from the backed-up dhaRti deck, writes the corrected dhaRti deck, then derives the
1517 deck from it. Every replacement must match exactly once, so nothing changes by accident.

    python fix_decks.py <folder holding python-pptx>
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from pptx import Presentation  # noqa: E402
from pptx.util import Pt  # noqa: E402

ROOT = Path("/home/sengar/dev")
SOURCE = ROOT / "old_decks" / "QUALYXUS_Pitch_Deck_dhaRti.pptx"
DEMO_URL = "https://drive.google.com/file/d/1ktDP_Sudm2bHiT3ViYH8cp2E7b9oSUZB/view?usp=sharing"


def paragraphs(slide):
    for shape in slide.shapes:
        if shape.has_text_frame:
            yield from shape.text_frame.paragraphs
        if getattr(shape, "has_table", False) and shape.has_table:
            for row in shape.table.rows:
                for cell in row.cells:
                    yield from cell.text_frame.paragraphs


def swap(slide, old, new, count=1):
    """Replace ``old`` inside single runs on this slide; it must occur exactly ``count`` times."""
    hits = 0
    for para in paragraphs(slide):
        for run in para.runs:
            if old in run.text:
                run.text = run.text.replace(old, new)
                hits += 1
    if hits != count:
        sys.exit(f"expected {count} match(es), got {hits}: {old[:70]!r}")


def cell_swap(slide, row, col, new):
    table = next(s for s in slide.shapes if getattr(s, "has_table", False) and s.has_table).table
    para = table.cell(row, col).text_frame.paragraphs[0]
    para.runs[0].text = new
    for extra in para.runs[1:]:
        extra.text = ""


def correct(prs):
    s = {i + 1: slide for i, slide in enumerate(prs.slides)}

    # 2. At a glance
    swap(s[2], "flagging process drift and machine wear before they cause scrap.",
         "tracking process drift and machine wear so causes can be fixed early.")
    swap(s[2], "1.8% calibrated false rejects, Modbus TCP / OPC UA cycle linking, advisory PLC output, 75 automated tests.",
         "~1% false rejects per part on a simulated six-camera station, machine-cycle linking, "
         "per-part PLC verdicts, 111 automated tests.")

    # 4. Solution
    swap(s[4], "Reject / alert flag; plant logic decides", "One verdict per part; plant logic decides")
    swap(s[4], "trains on a few hundred good parts", "trains on several hundred good parts")
    swap(s[4], "with a stated confidence bound, so the plant",
         "with a stated confidence bound, per part across all cameras, so the plant")
    swap(s[4], "Cycle drift flags process problems and component wear (check rings, tools, dies) before they "
               "cause scrap, with approval-gated fixes (simulation-validated).",
         "A wear index per component tracks drift (heater bands, check ring, cooling), with approval-gated "
         "fixes that cut scrap 79% in simulation.")
    swap(s[4], "Runs fully on-premise on an NVIDIA GPU; no cloud needed in production.",
         "Runs fully on-premise on an NVIDIA GPU, verified offline with no downloads.")

    # 5. Where we play
    swap(s[5], "The vision model learns any part from good samples; the cycle engine reads any machine over Modbus "
               "TCP / OPC UA, and the same signals track the wear behind the defects.",
         "Vision learns any part from good samples. Machine data comes over Modbus TCP / OPC UA (~20 Hz); "
         "fast signals need the high-speed capture on our roadmap.")
    swap(s[5], "Wear it tracks", "Wear it can track")
    cell_swap(s[5], 2, 4, "Vision + OPC UA ready; FOCAS + accelerometer next")
    for row in (3, 4, 5):
        cell_swap(s[5], row, 4, "Vision ready; fast capture + template next")
    cell_swap(s[5], 6, 4, "Vision + multi-camera ready; logical-defect checks next")

    # 6. Product
    swap(s[6], "a part passes only when every required view is inspected.",
         "a part passes only when every required view passes it. A bad image is a camera fault, not a reject.")
    swap(s[6], "writes a reject / alert flag to a standard PLC holding register over Modbus TCP.",
         "writes one verdict per part, with its part number and a heartbeat, to standard PLC registers over Modbus TCP.")
    swap(s[6], "Shadow mode, one-command rollback, drift alerts and approval-gated retraining.",
         "Shadow mode, one-command rollback, drift alerts, approval-gated retraining and password logins by role.")

    # 7. Built & measured
    swap(s[7], "1.8%", "~1%")
    swap(s[7], "False-reject rate on held-out good parts with conformal-calibrated thresholds (synthetic parts).",
         "False rejects per part, calibrated across a simulated six-camera station: 2 of 200 unseen good "
         "parts, with all 4 defect types caught (synthetic).")
    swap(s[7], "100% → 6.7%", "100% → 0%")
    swap(s[7], "False rejects on loosely fixtured parts after automatic part registration, with defect recall "
               "unchanged (synthetic).",
         "False rejects on parts rotated 8° after automatic part registration, with all defects still "
         "caught; 15% at 15° (synthetic).")
    swap(s[7], "129 / 129", "111")
    swap(s[7], "Audit records verified intact in the hash-chained, tamper-evident log; 75 automated tests in passing.",
         "Automated tests passing, plus end-to-end offline runs with every part judged correctly and the "
         "audit log intact (simulation).")
    swap(s[7], "Measuring the same metrics on real lines is the goal of the paid pilots.",
         "Nothing has run on a real PLC, Jetson or factory line yet: measuring that is the goal of the paid pilots.")

    # 8. Roadmap
    swap(s[8], "Multi-camera part verdicts; review & labelling loop",
         "Multi-camera part verdicts; camera-fault detection; review & labelling loop")
    swap(s[8], "Cycle engine (Modbus TCP, OPC UA), part-to-cycle linking, risk model and approval-gated fixes; "
               "MQTT / UNS publishing",
         "Cycle engine (Modbus TCP, OPC UA), multi-cavity part linking, risk model, machine-health index, "
         "approval-gated fixes; MQTT publishing")
    swap(s[8], "tamper-evident audit log, on-prem dashboard, advisory PLC output",
         "tamper-evident audit log, dashboard with logins, per-part PLC verdicts")
    swap(s[8], "TensorRT deployment and benchmarks on NVIDIA Jetson Orin",
         "TensorRT deployment, Jetson Orin benchmarks and industrial camera support")
    swap(s[8], "Offline installer with bundled models", "One-step offline installer")
    swap(s[8], "Supervised risk models trained on pilot data", "Early-warning and risk models trained on pilot data")
    swap(s[8], "Machine-health index per component + maintenance log", "High-speed signal capture; maintenance-log link")

    # 10. Competition
    swap(s[10], "Vendor-Led Setup per Part", "Under 45 Min per New SKU (Vendor Claim)")
    swap(s[10], "Inspection + Early Warning on Process Drift & Wear", "Inspection + Process Drift & Wear Tracking")
    swap(s[10], "1–2 Day Non-Invasive Install (No Machine Rewiring)", "Target: 1–2 Day Install (No Machine Rewiring)")
    swap(s[10], "Ranges are indicative and vary by application.",
         "Ranges are indicative. Others also link quality to process data (e.g. Kistler ComoNeo, Eigen, Sciemetric); "
         "our difference is learning from every inspected part, on-premise, at retrofit cost.")

    # 11. Business model
    swap(s[11], "enclosure, 1–2 day install.", "enclosure, target 1–2 day install.")


def delete_slides(prs, numbers):
    ids = prs.slides._sldIdLst
    entries = list(ids)
    for number in sorted(numbers, reverse=True):
        entry = entries[number - 1]
        prs.part.drop_rel(entry.rId)
        ids.remove(entry)


def build_1517(prs):
    s = {i + 1: slide for i, slide in enumerate(prs.slides)}

    swap(s[1], "dhaRti Foundation, IIT Dharwad · November 2026", "1517 Fund · Medici Project · November 2026")
    swap(s[2], "~1.7 lakh inspection points", "~170,000 inspection points")
    swap(s[2], "₹2–2.5L setup + ₹20,000 / month per line. Projected customer value ₹6–15L / line / year,",
         "~$2.5k setup + ~$230 / month per line in India. Projected customer value $7–17k per line per year,")
    swap(s[2], "Ask from dhaRti", "What a Medici Grant Funds")
    swap(s[2], "Incubation, pilot introductions and mentors, plus seed support: ₹5L to start, up to ₹15L over 9 months "
               "as milestones are met, to reach 3 pilot stations and a first paying customer.",
         "A bench test rig: an NVIDIA Jetson edge computer, an industrial camera and lighting, so the "
         "platform runs on real hardware and real parts for the first time.")
    swap(s[2], "Next 9 months: ", "Next: ")
    swap(s[2], "3 pilot stations in nearby plants, measured results, first paid conversion.",
         "real-hardware benchmarks on the test rig, then a first pilot in a nearby plant.")
    swap(s[15], "Company incorporation planned with dhaRti support.", "Company incorporation planned after the first pilot.")

    # The seed-support slide becomes the grant slide: same layout, new content.
    grant = s[17]
    swap(grant, "Seed Support Request", "What the Medici Grant Would Fund")
    swap(grant, "₹5 Lakh to Start; Up to ₹15 Lakh over 9 Months as Milestones Are Met",
         "Taking Qualyxus from Benchmarks to Real Cameras on Real Parts")
    swap(grant, "Use of Funds", "Bench Test Rig (~$1,000)")
    swap(grant, "Released Only as Milestones Are Met", "Why This Comes First")
    table_shape = next(x for x in grant.shapes if getattr(x, "has_table", False) and x.has_table)
    table = table_shape.table
    rows = [("Item", "Approx. cost"),
            ("NVIDIA Jetson Orin Nano developer kit (edge computer)", "~$400"),
            ("Industrial USB camera + lens", "~$400"),
            ("LED lighting, mounts and sample parts", "~$200")]
    for r, (item, cost) in enumerate(rows):
        cell_swap(grant, r, 0, item)
        cell_swap(grant, r, 1, cost)
    tbl = table._tbl
    for tr in list(tbl.tr_lst)[4:-1]:
        tbl.remove(tr)
    cell_swap(grant, 4, 0, "Total")
    cell_swap(grant, 4, 1, "~$1,000")
    for row in table.rows:
        row.height = Pt(54)
    table_shape.height = Pt(270)

    swap(grant, "Tranche 1 · ₹5L · Month 0", "The Gap Today")
    swap(grant, "Test rig, first station, company setup and patent filing.",
         "Everything so far runs on the public VisA benchmark and my own simulator.")
    swap(grant, "Milestone (month 3): rig live; Jetson speed and accuracy benchmarks published.",
         "Factories only trust numbers measured on real hardware.")
    swap(grant, "Tranche 2 · ₹5L · Month 3", "What the Rig Changes")
    swap(grant, "Two more stations and the intern's first 3 months.",
         "Real cameras, real lighting and real parts, on the edge computer a plant would use.")
    swap(grant, "Milestone (month 6): 3 pilot stations installed in nearby plants.",
         "It also becomes the station I carry into the first plant pilot.")
    swap(grant, "Tranche 3 · ₹5L · Month 6", "What It Proves (within ~8 weeks)")
    swap(grant, "Intern months 4–6, plant installs and travel, contingency.",
         "On-device speed and accuracy on real parts, published openly.")
    swap(grant, "Milestone (month 9): measured pilot results and the first paid conversion.",
         "A working station ready for the first pilot in a nearby plant.")
    swap(grant, "dhaRti's risk is capped at ₹5L at a time: ", "Costs are estimates: ")
    swap(grant, "each tranche follows a met milestone. Can be funded directly or through NIDHI-PRAYAS / SISFS with "
                "dhaRti as host incubator.",
         "Indian retail prices converted at ₹87 = $1.")

    closing = s[18]
    swap(closing, "Live demo and code walkthrough available on request", "Watch the demo video")
    for para in paragraphs(closing):
        for run in para.runs:
            if run.text == "Watch the demo video":
                run.hyperlink.address = DEMO_URL

    delete_slides(prs, [12, 13, 14, 16])


deck = Presentation(SOURCE)
correct(deck)
deck.save(ROOT / "QUALYXUS_Pitch_Deck_dhaRti.pptx")

deck = Presentation(ROOT / "QUALYXUS_Pitch_Deck_dhaRti.pptx")
build_1517(deck)
deck.save(ROOT / "QUALYXUS_Pitch_Deck_1517.pptx")
print("saved both decks")
