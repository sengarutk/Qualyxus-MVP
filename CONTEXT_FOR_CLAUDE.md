# Context for Claude: start here

Read this first, then `Qualyxus_Grant_Kit.md`. Together they carry the context of a long working session from 3–10 November 2026.

## What Qualyxus is

An on-premise edge-AI system for factories, built by Utkarsh Sengar, a fourth-year B.Tech CSE student at IIT Dharwad. It inspects every part with cameras, gives one pass or fail verdict per part to the plant's PLC, and links each defect to the machine cycle that made it, using machine data over Modbus TCP and OPC UA. It is a working prototype and **pre-revenue**. All results come from a public benchmark (VisA) and simulation. **Nothing has run on a real factory line, PLC, camera or Jetson yet.**

## Rules that matter most

1. **Never overstate traction or results.**
   - The founder has *spoken with the heads* of Rossell Techsys, Rotary Connectors and Moog about pilots. None is signed.
   - Rotary Connectors *agreed to help build the 360° inspector design* once funded.
   - There are no customers and no revenue.
2. **Older documents state projections as facts.** That covers pilots, ₹60L savings, a dhaRti grant and sub-8 ms speed, none of which happened. Never cite them. Files to avoid are listed below.
3. **Label every number:** benchmark, simulation, projection or plan.
4. **Emergent Ventures forbids AI help beyond proofreading.** The founder writes that application alone. Check every other programme's AI policy before drafting.
5. **Keep it concise and story-led.** The founder wants grant answers that sell a story, not technical detail, and fit within each form's character limit.

## Current files

| File | What it is |
|---|---|
| `Qualyxus_Grant_Kit.md` | **Source of truth for grants:** founder facts, story, traction, numbers, approved answers, status of each grant |
| `GradCapital_Application.md` | GradCapital answers (submitted) |
| `QUALYXUS_GradCapital.pdf` / `.pptx` | 10-page grant attachment, no dhaRti slides, clickable demo link |
| `QUALYXUS_dhaRti_Final_10min.pdf` / `.pptx` | 12-slide dhaRti presentation deck |
| `QUALYXUS_Pitch_Deck_dhaRti.pdf` / `.pptx` | Full 18-slide deck (backup for questions) |
| `QUALYXUS_QA_Prep.md` | About 140 likely questions with answers, a fill-in sheet, and phrases to avoid |
| `reports/` | Benchmark results. Only 4 of 12 VisA categories are done (`anomaly_benchmark_visa.md`) |

## Outdated: don't use or quote

- `QUALYXUS_Pitch deck.pdf`: older deck with projections stated as facts
- `QUALYXUS_Pitch_deck_v2.*` and `QUALYXUS_Pitch_deck_8min.*`: superseded
- `QUALYXUS_Pitch_Deck_1517.*`: the earlier 1517 version (111 tests, no traction slide)
- `old_decks/`: backups and the scripts that built the decks

## Software status (branch `feat/process-quality-intelligence`)

- 169 automated tests pass, and the end-to-end six-camera simulation passes: `python scripts/smoke_360.py`
- **Recent fixes**, found by deliberately breaking the system:
  - The PLC heartbeat now stops if cameras die.
  - A verdict can't be lost to a full disk or an alert storm.
  - Parts no camera saw get a "not inspected" verdict.
  - The audit log survives power cuts and restarts.
  - Model files are signed.
  - The API is local-only by default.
  - YOLO (AGPL) is optional.
  - Damaged databases are set aside instead of stopping the line.

## Still to do

1. **Full VisA benchmark (all 12 categories).** Run with `reports/visa_full/run_all.sh`. Dinomaly training on the laptop GPU crashed Windows' WSL service 5 times on 8 November, and Blender was also using the GPU at the time. Options: run on a free cloud GPU (Colab or Kaggle), update WSL with `wsl --update` and retry with nothing else using the GPU, or run natively on Windows.
2. **Long soak test** (30+ minutes) and a **real Docker image build:** both started, never completed.
3. **Python 3.11+ and newer PyTorch,** to clear about 64 dependency advisories.
4. **Open grant decisions:**
   - Record the Loom for 1517 Medici.
   - Apply to Emergent Ventures (founder writes it), Magnificent Grants and Inflection Grants.
   - Decide which budget pays for the 360° design.
   - Confirm the advisors and the three companies agree to be named.
