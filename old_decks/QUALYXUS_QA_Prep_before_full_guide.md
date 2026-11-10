# Qualyxus — Q&A Prep for the dhaRti Pitch

**How to answer:**
- Keep each answer to 20–30 seconds: lead with the answer, then one fact, then stop.
- If you don't know, say: *"I don't know yet. Here's how I'll find out: …"*
- Never stretch a number. Every figure below is one you can defend.

★ = very likely to be asked. Prepare these out loud.

---

## 1. Traction & validation (the toughest area)

**★ Do you have any customers or pilots right now?**
> Not paying customers yet. I've spoken with the heads at Rossell Techsys, Rotary Connectors and Moog about piloting on their parts, and Rotary Connectors has agreed to help build the 360° inspector design once we're funded. None of these is a signed pilot yet; this round is what turns them into real-plant pilots.

**★ Has this ever run in a real factory?**
> No. Everything so far is on a public industrial benchmark (VisA) and on our own simulator. That's why tranche 1 is a lab test rig and tranche 2 is three plant stations. Real-line measurements are the month-9 milestone.

**Are the companies on the traction slide real customers?**
> Not customers. I've spoken with their heads about pilots, and nothing is signed yet. Rotary Connectors has agreed to help with the 360° design. The ₹35 lakh figure in the full deck is modelled for example lines, not for these companies.

**What exactly did Rotary Connectors agree to?**
> *(Say only what was actually agreed: who you spoke to, what they will contribute, whether it's in writing, and whether it's free or paid.)*

**Which part of the budget pays for the 360° design?**
> *(Decide this before you present. The ₹15L table has no separate 360° line, so say whether it comes from the test-rig budget or Rotary Connectors' contribution covers it.)*

**How do you know plants want this?**
> The pain is well documented. Indian auto-component suppliers face OEM penalties per defective part, and plants pay ₹20–35k a month per inspector per shift. The paid pilot is how I'll test willingness to pay. A plant that pays ₹1L for a pilot has shown real intent.

**What's the one thing that would prove you wrong?**
> If plants won't pay ₹20k a month after a successful pilot, or won't share machine data. Both get tested in the first 9 months, which is why the dhaRti money is milestone-based.

---

## 2. Technology

**★ What exactly is the AI? Did you build the models?**
> I use state-of-the-art open models: Dinomaly and PatchCore on Meta's DINOv2, and RF-DETR for defect outlining. All have licences that allow commercial use. What I built is everything around them, which is what makes it work in a factory:
> - automatic part alignment and tiling
> - statistically calibrated reject thresholds
> - linking each part to its machine cycle
> - the risk model
> - model operations: shadow mode, rollback, audit log
> - the PLC and dashboard integration

**★ What does AUROC 0.95–0.99 mean in simple terms?**
> It's how well the model separates good parts from defective ones: 1.0 is perfect, 0.5 is a coin flip. We scored 0.95 to 0.99 on four product categories of VisA, a public industrial defect benchmark. The older standard method scored 0.56 to 0.96 on the same categories.

**★ So what percentage of defects do you actually catch?**
> It depends on the part. At a threshold set to keep false rejects low, the benchmark model caught 90% of defects on circuit boards and only 35% on the hardest category (capsules), across four categories of the public VisA benchmark. On our simulated parts it caught all four defect types. That's why we start as a **second check alongside inspectors**, not a replacement, until a real line proves the catch rate.

**What is "calibrated false rejects"?**
> Most AI tools pick a threshold by eye. We set it using a statistical method (conformal calibration) on held-out good parts. It gives a stated bound on how many good parts will be rejected, so the plant knows its false-reject rate before go-live. On our synthetic parts it measured 1.8%.

**Why does the false-reject rate matter so much?**
> Because that's what gets AI switched off in factories. If you reject 5% of good parts, operators lose trust within a week.

**How many good parts do you need to train on a new part?**
> Several hundred, and no defect samples are needed for detection. With 300 parts our lighter model scored AUROC 0.946, against 0.960 with the full set. Calibrating needs more good parts on top: a six-camera station needs at least 600. Competitors like Siemens Inspekto claim 20–30 samples, so reducing this is on my list.

**What about defects you've never seen before?**
> That's the strength of learning only from good parts. Anything that looks unusual gets flagged, even a defect type nobody has seen. Naming the defect type needs labelled examples, and the review queue collects those over time.

**How long does training take, and does it need the internet?**
> It trains on a GPU workstation on-site: minutes for PatchCore, longer for Dinomaly. I'll publish exact timings with the Jetson benchmarks. No internet is needed: the models are bundled, and I tested that on an offline setup with an empty download cache.

**★ What happens when lighting changes, the part moves, or the camera gets dirty?**
> Part alignment handles position. On parts rotated 8° it cut false rejects from 100% to 0% without losing any defects, and 15% remain at 15° (simulated parts). A blurred or noisy image is reported as a camera fault, not a bad part. Drift detection flags when images change, and recalibration or retraining goes through an approval step. Proper fixed lighting in an enclosure is part of the station design.

**How fast is it?**
> Defect outlining measured 16.5 ms per image on a laptop GPU. I haven't benchmarked on Jetson yet; that's the month-3 milestone. Most parts take seconds to make, so even a few hundred milliseconds is real-time per part.

**★ What does "linking each defect to its machine cycle" actually mean?**
> When the camera sees a defect, the machine that made the part has usually moved on. So we match each part to the exact production cycle that made it, using a part ID, a counter, or a measured delay. That cycle's sensor curves (pressure, temperature, spindle load) then show what went wrong. Over time we learn which signal patterns cause which defects.

**How do you get machine data?**
> From standard industrial protocols: Modbus TCP and OPC UA, which most modern machines and PLCs expose. CNC-specific connectors (MTConnect, FANUC FOCAS) are in the next 6 months. For old machines with no data port, we add cheap sensors, like a ₹5–15k accelerometer.

**Can you predict defects before they happen?**
> Not reliably yet, and I won't claim it. In simulation the risk model warned before only 16 of 60 defect episodes. The stronger result is the approval-gated setpoint fixes, which cut scrap by 79% in simulation. Improving early warning on real pilot data is a main goal.

**Why not just use ChatGPT or a cloud AI API?**
> Factories, especially defense suppliers, often can't send images outside the plant, and the line can't wait on a network round-trip. Everything runs on-premise.

**What edge hardware do you use, and why NVIDIA?**
> A Jetson Orin-class edge computer, with a GPU workstation for training. NVIDIA has the best-supported toolkit for fast industrial AI (TensorRT), and the Jetson is cheap enough to fit our ₹2–2.5L station price.

**Is your code tested? Is it production-grade?**
> 166 automated tests pass, including forced failures like dead cameras, a PLC outage and power cuts, plus an end-to-end offline run. It has a model registry with shadow mode and one-command rollback, and a tamper-evident audit log. It's production-style engineering, but it hasn't run in production yet.

---

## 3. Safety, data & regulation

**★ What if your system wrongly stops a line, or misses a critical defect?**
> It's **advisory only**. It writes one verdict per part, with the part number and a heartbeat, to PLC registers, and the plant's own PLC logic decides whether to divert or stop. If a camera stops reporting, the part is marked not inspected, never passed. I've tested this over a real network connection, but not on a real PLC yet. We never touch certified safety functions. During pilots it runs in shadow mode next to the inspectors, so nothing changes on the line until the plant trusts the numbers.

**Does this need safety certification (ISO 13849)?**
> It's designed to stay outside the safety-related control system, so existing safety validation stays intact. We confirm that with each plant's safety engineer.

**Where does the data go? What about data security?**
> Nowhere. Everything runs on-premise with no cloud dependency. That matters a lot for defense and aerospace suppliers.

**What's the audit log for?**
> Every inspection verdict is recorded in a hash-chained log, so any tampering is detectable. Quality auditors for automotive (IATF 16949) and aerospace (AS9100) ask for this kind of traceability.

**Who's liable if a bad part escapes to a customer?**
> The plant's quality system stays responsible. We're an additional check, and contracts will cap our liability, as is standard for inspection vendors. In pilots we run beside humans, not instead of them.

---

## 4. Market & competition

**★ How did you get the $590M SAM?**
> Bottom-up, about 1.7 lakh inspection points:
> - CNC machining: ~70k, from 3–4 lakh installed CNC machines in India
> - Injection moulding: ~30k
> - Casting and forging: ~15k, from ~5,000 foundries and 500+ forging units
> - Stamping, welding and assembly: ~15k
> - Southeast Asia: ~40k
>
> 1.7 lakh points × about ₹3L a year ≈ ₹5,100 Cr, or $590M. These are estimates; no official census exists.

**Isn't the $45B TAM inflated?**
> It's two published 2025 estimates added together: AI inspection (~$31B) and predictive maintenance (~$14B). Definitions overlap, so treat it as an upper bound. The number I'd defend is the bottom-up SAM.

**★ How are you different from SwitchOn?**
> SwitchOn is a strong, well-funded Indian AI-inspection company: $4.2M Series A in 2023 and $8M pre-Series B in July 2026. That validates the market. Our angle is linking each defect to the machine cycle and the wear behind it, so we sell to both the quality and the maintenance budget. We also lead with a calibrated false-reject rate and run fully on-premise.

**Why won't Cognex or Keyence just do this?**
> They sell cameras, typically ₹1.2–7.5L each plus integrator programming per part. Their model is hardware plus integrators, not continuous learning tied to machine data. They could add it, so our moat has to be speed, price, and the process templates and data we build plant by plant.

**Other competitors you know of?**
> - **Indian AI inspection:** SwitchOn, Qualitas Technologies, Jidoka.
> - **Global AI inspection:** Landing AI, Instrumental, Elementary.
> - **Machine monitoring:** Infinite Uptime and Nanoprecise. They put vibration sensors on motors and pumps; we focus on production machines, where wear shows up as defects.

**★ What's your moat? Can't someone copy this?**
> The models are open, so the moat isn't the model. It's:
> 1. The integration: alignment, calibration, part-to-cycle linking and model ops, which took a long time to get right.
> 2. Process templates that grow with every plant.
> 3. A data flywheel: every pilot teaches us which machine signals cause which defects.
> 4. Switching cost once it's wired into a plant's PLC and quality records.
>
> I'll also assess patents on the part-to-cycle linking and calibration methods.

**★ What if SwitchOn or Qualitas add machine-data linking? What's your head start, and what's special about you?**
> 30-second answer: "They could, and they have more money and more customers than me. My head start isn't the code, it's the focus. For them, linking machine data is a side feature that needs integration work with each machine and a different buyer, the maintenance head. For me it's the whole product. My plan is to be the specialist they'd rather partner with or buy than copy. And my costs are so low that I can serve the small plants their sales model can't reach."
>
> If they push: be honest that the head start is months, not years. What turns months into a lasting edge:
> - building process knowledge plant by plant
> - pilots in the Hubballi–Dharwad–Belagavi cluster
> - running fully on-premise, for defense suppliers who can't use cloud tools
>
> **Don't say:** "They can't do it" or "nobody else does this."

**Why now?**
> Three things changed. Foundation vision models like DINOv2 made learning from good parts only work well. Edge GPUs got cheap. And Indian manufacturing is booming: auto components hit ₹7.6 lakh crore in FY26, up 12.7%, with global OEMs shifting sourcing to India and demanding lower defect rates.

**Why India first?**
> The suppliers are here and growing fast. It's close to campus, so pilots are cheap to run. Prices are lower, but software margins are still 80%+. Once we have Indian references, we go to Southeast Asia at about 1.5–2× the price, then the US and EU at about 4×.

---

## 5. Business model & pricing

**★ Why ₹20,000 a month? Isn't that too cheap, or too expensive, for Indian plants?**
> It's anchored to the alternative. One inspector costs ₹20–35k a month with benefits, per shift. We cost less than one inspector and cover all three shifts. A line gets ₹6–15L of value a year, so the plant pays back in 4–10 months. The paid pilot is where we test whether that holds.

**Where does the ₹6–15L value per line come from?**
> - **Inspection labour:** 50–100% of one inspector across three shifts, about ₹3.5–12L.
> - **Scrap:** 0.5–1% of a line's ₹1–3 Cr output, about ₹0.5–3L.
> - **Escaped defects:** avoided customer penalties and returns, which vary a lot by customer.
>
> These are assumptions to validate in pilots.

**What does the hardware cost you?**
> About ₹1.5–1.8L: Jetson ₹40–50k, industrial camera and lens ₹40–80k, lighting ₹15–30k, enclosure and wiring ₹15–25k, install about ₹20k. We sell it at ₹2–2.5L, a 20–30% margin. The software carries the real margin, 80%+.

**Will plants actually pay a subscription?**
> Some Indian plants prefer one-time purchases. If needed we can offer a 3-year prepaid licence, which is the same economics with a different label. The pilot will tell us.

**What's your customer acquisition cost?**
> Our target is ₹5–8L per plant: the pilot plus field sales. At 6 lines a plant, gross profit is about ₹11.5L a year, so we recover the acquisition cost in 5–9 months.

**How will you scale installs if every plant is different?**
> Process templates mean each new process is configuration, not new code. After that, machine distributors and automation integrators install for us, for a 20–25% margin.

---

## 6. Predictive maintenance

**★ How is your predictive maintenance different from vibration-monitoring companies?**
> They monitor motors and pumps for breakdowns. We watch production machines and flag the wear that's starting to affect parts, such as check rings, cutting tools, dies and welding tips. We see both the machine signal and the part quality, so we know when wear actually matters.

**Can you predict when a machine will fail?**
> Not yet, and I won't claim it. We warn when wear starts affecting quality, so it's fixed in the next planned stop. Remaining-life estimates need failure history from real plants; that's on the 12–18 month roadmap.

**Is predictive maintenance in your revenue numbers?**
> No. It's priced as an add-on from ₹5,000 per machine per month, starting in the pilot phase, and left out of the projections as upside.

---

## 7. Team & commitment

**★ You're a student. How will you manage studies and a startup?** *(answer honestly, in your own words)*
> Tell them your actual plan: weekly hours, semester timeline, and what happens after graduation. The ₹15L includes an intern or junior engineer so progress doesn't depend only on your hours.

**★ You're a solo founder. Who else is on the team?**
> Right now it's me, plus advisors from aerospace and defense manufacturing. My biggest hire is a co-founder with factory or industrial-sales experience, and I'd value dhaRti's network for that. The plan also funds an engineering intern from month 3.

**What do your advisors actually do? Have they agreed?**
> *(Answer only what's true: how often you speak, and what each one helps with.)* If asked about Dr. Rakesh Kumar, say upfront that he works at Rossell Techsys, which is also a target pilot company.

**Why are you the right person to build this?**
> I've already built the full platform end to end: detection, calibration, part-to-cycle linking, risk models and model operations. I've also taught myself how these processes create defects, including building a simulator of an injection-moulding machine.

**Did you build all of this yourself?**
> *(Be honest about tools and help, e.g. "I designed and built it, using open-source models and AI coding tools to move fast.")* Make sure you can explain every component if they dig in.

---

## 8. The ask & dhaRti

**★ Why ₹15 lakh? What if we can only give less?**
> ₹15L funds exactly one step: getting from simulation to measured results in real plants.
> - ₹5L for 3 pilot stations
> - ₹1.5L for a test rig
> - ₹3L for an intern
> - ₹1.5L for installs and travel
> - ₹1.5L for company setup and a patent filing
> - ₹1L for compute
> - ₹1.5L contingency
>
> It's released in three tranches tied to milestones. With ₹5L I can still start: the rig, the first station and company setup.

**★ What will you achieve, and by when?**
> - **Month 3:** test rig running, and Jetson speed and accuracy benchmarks published.
> - **Month 6:** three pilot stations installed in nearby plants.
> - **Month 9:** measured results and the first paying customer.

**★ Do you want equity investment or a grant? What would you give in return?** *(Don't commit on the spot)*
> "I'd like to understand dhaRti's standard terms, whether it's a grant, a convertible or equity, and I'm open to the usual incubator arrangement." Then decide later.

**Why should dhaRti back this over other startups?**
> It fits dhaRti's Industry 4.0 and defence-technology focus. It's already built rather than an idea. The ask is small and milestone-gated. And the first pilots are right here in the Hubballi–Dharwad–Belagavi cluster, near the foundries and the Aequs aerospace zone.

**Is the company incorporated? Who owns the IP?** *(check these facts before the meeting)*
> Incorporation is planned with dhaRti's support, and it's 100% founder-owned. On IP: "I'm happy to follow IIT Dharwad's and dhaRti's IP policy." Check whether any of the work was done on institute resources or for coursework.

**Have you applied to other programmes or grants?**
> *(Be truthful.)* Next on the plan are Startup India Seed Fund and NIDHI-PRAYAS through dhaRti, then a ₹2 Cr pre-seed once pilots show results.

**What happens with the ₹2 Cr pre-seed?**
> It buys 18 months: 3 engineers, field sales, 10 pilot stations, and IP and compliance. The goal is 40+ paying lines and about ₹1 Cr in software revenue a year. We raise it only after the pilots prove the model works.

---

## 9. Risk & what-if

**★ What's your biggest risk?**
> That accuracy on real factory parts falls short of what plants need, or that plants won't share machine data. Both get tested in the first 9 months. If machine data is hard to get, the inspection product still stands alone, and the accelerometer route covers older machines.

**What if the pilot fails?**
> Then I'll know exactly why: catch rate, false rejects, data access, or price. The milestone structure means dhaRti's money stops at the last tranche reached, not after ₹15L is spent blindly.

**What if a big company enters?**
> Big players move slowly in Indian SME plants and sell at higher prices. Our edge is low cost, fast installs, and process knowledge built plant by plant.

**What would make you quit or pivot?**
> If, after the pilots, plants see value but won't pay ₹20k a month, I'd test a lower-cost vision-only product. If machine data is consistently blocked, I'd go vision-first with add-on sensors.

---

## 10. Curveballs

- **"Explain this to my grandmother."** → "A camera that spots bad parts, and then tells the factory which part of the machine caused it, so they can fix it before making more."
- **"What did you learn that surprised you?"** → False rejects, not accuracy, kill factory AI; and defects are created at the machine but only seen later at the camera.
- **"Why should we believe your numbers?"** → "Every number is labelled benchmark, simulation or projection. The pilots replace the projections with measurements."
- **"What would you do with ₹50 lakh instead?"** → "Same plan, more pilot stations at once, and a co-founder hire. But ₹15L is enough to prove it first."

---

## Numbers to remember

| Item | Number |
|---|---|
| Benchmark detection score (AUROC) | 0.95–0.99 on VisA (4 categories) |
| False rejects | ~1% per part target; 2 of 400 unseen parts on a simulated six-camera station |
| Alignment gain | False rejects 100% → 0% at 8° rotation, no defects lost (15% at 15°) |
| Speed | 28 ms per image, 170 ms per six-view part for the vision models, plus roughly 75 ms to save audit records (laptop GPU and disk; Jetson not yet tested) |
| Price | ₹2–2.5L station + ₹20k / month per line |
| Value per line | ₹6–15L a year; payback 4–10 months |
| Market | TAM $45B · SAM $590M · SOM $5.5M (2,000 lines) |
| Plan | 20 → 90 → 300 lines; ARR ₹0.48 → 2.1 → 7.0 Cr |
| dhaRti ask | ₹15L in 3 × ₹5L tranches over 9 months; ₹5L minimum |
| Next round | ₹2 Cr pre-seed, 18 months, 40+ lines, ~₹1 Cr ARR |
