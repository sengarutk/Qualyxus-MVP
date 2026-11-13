# Qualyxus Grant Kit

Everything needed to keep applying for grants. Upload this file to Claude Cowork, then say: "Use this kit to help me with my next grant application."

---

## 1. Rules for every application

- **Never overstate.** Qualyxus is **pre-revenue**, with **no signed pilots** and **no customers**. Results come from a **public benchmark and simulation only**. Nothing has run on a real factory line, PLC, camera or Jetson yet.
- **Label every number** as benchmark, simulation, projection or plan.
- **Older documents contain projections written as facts.** Never use them. That covers pilots at Rossell, Magellan and Rotary, ₹60L savings, a dhaRti grant, sub-8 ms speed and 4-hour installs, none of which happened.
- **Check each programme's AI policy before drafting.** Emergent Ventures rejects applications that used AI for anything except proofreading, so the founder writes that one alone.
- **If a form asks about other applications, answer truthfully.**

---

## 2. The founder

- **Name:** Utkarsh Sengar
- **Education:** B.Tech CSE, IIT Dharwad, fourth year
- **Email:** cs23bt026@iitdh.ac.in · **Phone:** +91 8792356189
- **LinkedIn:** linkedin.com/in/utkarsh-sengar (an older deck says /in/sengarutkarsh; check which is right)
- **Experience and achievements** (as stated by the founder):
  - AI intern at **Rossell Techsys** and **Moog**, applying AI in industry
  - Competed for IIT Dharwad at the **Inter IIT Tech Meet** on the AI problem statement, and performed strongly (add a rank if one exists)
  - Active research on **AI for drone swarms**, improving performance and saving energy; papers in preparation
  - Did well in **debating and MUN**: intra-college and at Inter IIT
- **Company:** not yet incorporated, 100% founder-owned
- **Advisors** (confirm each has agreed to be named):
  - Dr. Rakesh Kumar: VP R&D, Rossell Techsys (ex-Moog, ex-Indian Navy). Rossell is also a pilot prospect, so disclose this if asked.
  - Shankar Lal: Discipline Chief, Cyient (21 years at HAL)
  - Ramesh Rana: Operations Lead, SIG SAUER

---

## 3. The story in one line

Qualyxus: AI that catches bad parts on the factory floor and tells the plant why they went bad.

**Core insight:** factories run two systems that never talk. Cameras say a part is bad, but not why. Machines know how each part was made, but never learn which parts failed. Qualyxus connects them, entirely on-premise.

---

## 4. Traction (exact wording, don't strengthen it)

- **Spoken with the heads** of Rossell Techsys, Rotary Connectors and Moog about testing Qualyxus on their parts. These are conversations, not signed pilots.
- **Rotary Connectors has agreed to help build the design of the 360° inspector**, starting once funded.
- **Not settled yet:** what exactly Rotary Connectors agreed to, whether it's in writing, and which budget pays for the 360° design.

---

## 5. Numbers you can use

| What | Number | Type |
|---|---|---|
| Anomaly detection AUROC | 0.95–0.99 on 4 of 12 VisA categories | Benchmark |
| Defects caught at strict false-reject settings | 35–90%, by category | Benchmark |
| False rejects per part (six-camera station) | ~1% target; 2 of 400 unseen good parts | Simulation |
| Defect types caught | All 4 | Simulation |
| Part alignment | False rejects 100% → 0% at 8° rotation; 15% at 15° | Simulation |
| Defect outlining | Mask IoU 0.83; 16.5 ms per image on a laptop GPU | Simulation |
| Scrap reduction from approved setting changes | 79% | Simulation |
| Early warning of defects | 16 of 60 episodes (weak; don't claim prediction) | Simulation |
| Automated tests | 169 passing, including deliberate failures | Measured |
| Price | ₹2–2.5L setup + ₹20k / month per line | Plan |
| Value per line | ₹6–15L a year; payback 4–10 months | Projection |
| Market | TAM $45B; SAM $590M; SOM $5.5M | Estimate |
| dhaRti ask | ₹15L in 3 milestone tranches of ₹5L over 9 months | Plan |

**Planned use of ₹15L:**

| Item | Amount |
|---|---|
| 3 pilot stations | ₹5L |
| Lab test rig and Jetson kit | ₹1.5L |
| Engineering intern, 6 months | ₹3L |
| Plant installs and travel | ₹1.5L |
| Incorporation, DPIIT recognition and provisional patent | ₹1.5L |
| Compute | ₹1L |
| Contingency | ₹1.5L |

**Milestones:**

- **Month 3:** test rig live and Jetson benchmarks published
- **Month 6:** three pilot stations installed
- **Month 9:** measured results and the first paid conversion

---

## 6. Approved answers (GradCapital, each under 1,000 characters)

**Describe your idea / project**

> Every day, factories make parts that are already bad, and nobody knows until hours later, after the metal, machine time and labour are gone. The camera at the end of the line can say a part is bad, but never why. The machine that made it knows exactly what went wrong, but never learns which parts failed. Two systems, side by side, never talking.
>
> Qualyxus connects them. A small AI box at the line checks every part and traces each defect back to the moment the machine made it, so the plant fixes the cause instead of sorting scrap. It runs fully offline, because defence and aerospace suppliers can't send images to the cloud.
>
> I built the whole platform myself, and it already works on a public industrial benchmark. I've spoken with the heads of Rossell Techsys, Rotary Connectors and Moog about testing it on their parts, and Rotary Connectors has agreed to help build our 360° inspector. This grant puts Qualyxus in front of real parts for the first time.

**How did you spend your time in school and college?**

> I've spent college where AI meets the physical world.
>
> As an AI intern at Rossell Techsys and Moog, I worked on bringing AI into real industry, where data is messy and mistakes cost real money. That pushed me to build Qualyxus end to end. Before calling it ready, I deliberately broke it: killed cameras, cut the power, unplugged the factory controller. Then I fixed every weakness it exposed.
>
> I represented IIT Dharwad at the Inter IIT Tech Meet on the AI problem statement, and performed strongly. I'm also doing active research on AI for drone swarms, helping them perform better while using less energy, and I'm preparing it for publication.
>
> Outside the lab, I debate. I've done well in college debates and MUNs and at Inter IIT, which taught me to think on my feet, take hard questions and argue from evidence. It's the same skill I use when I pitch Qualyxus to the people who run factories.

---

## 7. Where each application stands

| Programme | Status | Notes |
|---|---|---|
| GradCapital | **Submitted** | $1k–$5k grant; the $60k grant was declined earlier |
| 1517 Medici | To do | Record the Loom first (4–5 minutes, face and screen) |
| Emergent Ventures | **Submitted** (13 Nov 2026) | Asked $35,000 in two parts: $17k core (test rig, 3 pilot stations, engineer, installs) + $18k extension (360° inspector, 6 months' living costs after graduation, engineer months 7–12). Attachment: QUALYXUS_EmergentVentures.docx; demo video in the Multimedia URL |
| Magnificent Grants | To do | Age 25 or under, $10k+, no need to leave college. Check its AI policy first |
| Inflection Grants (Edge City) | To do | Under 25, $2k, rolling. Check its AI policy first |
| Bagel Fund, Narisetti Grants | Optional | Small, quick |
| Exception Raised, evm.capital, Kothari | Check eligibility | |
| OSV Fellowship | Next cycle | 2026 closed on 30 April; reopens around January 2027 |
| Thiel, 776 | Not now | Both expect you to leave college; 776 is climate-only |
| Non-trivial, Fast Grants, Impetus, Unitary | Skip | Wrong age or field |

---

## 8. Files and links

- **Grant PDF (10 pages, no dhaRti slides):** QUALYXUS_GradCapital.pdf
- **dhaRti presentation deck (12 slides):** QUALYXUS_dhaRti_Final_10min.pdf
- **Full deck (18 slides):** QUALYXUS_Pitch_Deck_dhaRti.pdf
- **Q&A prep (about 140 questions):** QUALYXUS_QA_Prep.md
- **Demo video:** https://drive.google.com/file/d/1ktDP_Sudm2bHiT3ViYH8cp2E7b9oSUZB/view (check it opens the demo and is shared as "Anyone with the link")
- **Website:** https://qualyxus-web.vercel.app (being fixed separately; check it matches these facts before linking it)
