[← Verityne](../README.md)

> Detector 6 in full: a real keystroke corpus, the feature-set ablation, and the attack that beats it.

# Detector 6, measured

Both halves of this corpus are real, and that is the only reason the numbers
below are worth printing.

**The genuine half** is 12,000 typing sessions drawn from the
Aalto 136M keystroke study - 168,595 people typing in a browser, with press and
release timestamps per key - plus 2,040 sessions from the
CMU Killourhy-Maxion benchmark's 51 subjects. **The automated half** is
3,000 runs of a real headless Chromium, driven through the
automation APIs a kit actually uses and recorded by the *same collector this app
ships* (`frontend/lib/telemetry.ts`, compiled, not reimplemented). Nothing here
was written by us to look like what we expected it to look like.

| | |
| --- | --- |
| Held-out ROC-AUC, subject-disjoint | **1.000** |
| Recall at the operating threshold | **100%** |
| Genuine humans flagged | **0.07%** |
| Worst unseen strategy | **`replay_human` at 0.500** |

The split is **subject-disjoint**: no participant appears on both sides. Keystroke
dynamics is a biometric - people are individually identifiable from it - so a
row-wise split would have measured whether the model can recognise a person, not
whether it can recognise automation. The threshold is read off held-out humans
for a stated 1% false-positive budget, the same way the
linkage threshold is chosen from a false-link budget, rather than left at 0.5.

648 hyperparameter configurations were searched, ranked **by
transfer to a human population never fitted on** rather than by held-out AUC.
Ranking on AUC would have been meaningless: every configuration in the grid
reaches 1.000 there, so the search would have picked one at random and called it
tuned. Fitting is seconds of compute - the expensive part of this detector was
acquiring data neither half of which we wrote.

#### The first fit was reading our own preprocessing

Its highest-gain feature was `n_keys`, at 0.435 - session length. Human sessions
are accumulated to a target length and automated ones fill a fixed five-field
form, so length was an artefact of how the corpus was *chopped*, and the model
had found it immediately. Two more features were nearly as bad: `backspace_rate`
(Aalto participants make typos, our automation never does, and the CMU corpus
contains only clean entries) and `pause_rate` (an artefact of Aalto's
sentence-by-sentence protocol).

That model scored a held-out AUC of 1.000 and flagged **93% of CMU's
participants** as bots. Dropping `n_keys` and splitting the rest into features
grounded in motor physiology versus features describing the task is what the
`core` / `core+context` ablation measures:

| Feature set | Features | Held-out AUC | False positives on a human population never fitted on |
| --- | --- | --- | --- |
| **`core`** - dwell/flight distribution shape, rollover, quantisation | 22 | 1.000 | **38.8%** |
| `core+context` - plus backspaces, pauses, typing speed | 25 | 1.000 | 81.6% |

Identical held-out AUC; twice the false-positive rate on strangers. `core` ships.
`n_keys` is in neither set and is not computed into either.

That remaining 39% is
the honest ceiling on this detector, and it is stated rather than buried: CMU's
subjects typed one memorised password four hundred times on a lab rig, which is
about as far from a merchant filling a KYC form on a phone as a human typing
corpus gets. It is the hardest transfer test available to us and the number is
not good. An Indian KYC queue is a third population again, and nothing here
measures it.

#### Generalising to automation it has never seen

Each row trains on every strategy but one and tests on the one held back, because
the kit in production next month is not in this corpus:

| Unseen strategy | ROC-AUC | Recall | False positives |
| --- | --- | --- | --- |
| `replay_human` | 0.500 | 0% | 0.0% |
| `type_gaussian` | 0.998 | 17% | 0.0% |
| `cdp_raw` | 1.000 | 100% | 0.0% |
| `type_fixed` | 1.000 | 100% | 0.0% |
| `type_jitter` | 1.000 | 100% | 0.0% |
| `type_no_delay` | 1.000 | 100% | 0.0% |

`type_gaussian` is worth reading twice: it ranks almost perfectly
(0.998) but only
17% of its sessions clear the
threshold. A competent attacker with plausible gaussian delays is *separable* but
not *separated* at an operating point chosen to protect genuine users - which is
what a false-positive budget costs, and why it is quoted next to the recall
rather than in a footnote.

`fill_value` - setting `input.value` directly - is absent from this table because
it emits **no key events at all**. There is no rhythm to model, so its 500
sessions are excluded from the keystroke corpus entirely. It is caught by the
rules instead (an empty keystroke buffer, a straight pointer path, a sub-15s
fill), which is what the rule layer is still there for.

#### The attack that beats it, which we built ourselves

`replay_human` scores **0.500** - chance. It takes a real
Aalto session and replays that person's exact dwell and flight timings through
the devtools protocol, including the key rollover that every other strategy is
structurally unable to produce, by laying the presses and releases on a timeline
and dispatching in time order rather than key by key.

This is not a bug to be fixed by a better model. The rhythm genuinely *is* human,
so no model of rhythm can separate it, and the honest way to report Detector 6 is
that it raises the cost of automating a KYC form from *free* to *you must first
record a real human filling one*. Against a fraud kit that has also hidden its
`navigator.webdriver` flag and randomised its field order, the full detector's
score on these sessions falls from 0.96 to 0.30.

What still catches it is that a replayed recording is a **reused** one - the same
timings arriving under many identities - which is the linkage problem this repo
already solves for faces and files, not a timing problem. That is the roadmap
item, and it is stated here rather than implied by a number that does not exist.

---

---

[← Verityne](../README.md) - [Architecture](architecture.md) · [Corrections](corrections.md) · [Results](results.md) · [Measured on real data](real-data.md) · **Detector 6** · [Fusion & policy](fusion.md) · [What it proves](evaluation.md) · [API & config](api.md)
