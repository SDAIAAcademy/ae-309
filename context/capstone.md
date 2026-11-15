---
course: "Computer Vision Engineering"
code: "AE-309"
version: "2027.1.0"
deck: "capstone"
day: 5
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-309/slides/capstone.html"
---

# Computer Vision Engineering

## Capstone: Capstone: a service that meets its budget, defended on a failure

An inspection or counting service for Al-Manar with its field-conditions test set, model card and release memo

#### [capstone#title] Capstone: a service that meets its budget, defended on a failure

*SDAIA Academy · Computer Vision Engineering*

Capstone: An inspection or counting service for Al-Manar with its field-conditions test set, model card and release memo

AI Engineering Track · 4 teaching days + capstone day · 8 min talk and demo · 4 min Q&A · AE-309

#### [capstone#agenda] How the day runs

*Capstone*

- **The brief, the rubric and final prep**: what is submitted, how it is scored, a last rehearsal
- **Defences**: eight minutes of talk, then four minutes on a failure case the assessor picks
- **Review and closing**: peer feedback, the trainer's review, what next

#### [capstone#the-brief] A service for Al-Manar that meets a budget you stated

*The brief*

- **Inspection.** Find potholes and patches, or segment cracks, in road frames. Accuracy is average precision or mask overlap, by lighting and district.
- **Counting.** Count vehicles across a line in traffic clips. Accuracy is count error per clip, with identity switches reported beside it.

> **You wrote the budget in lab 4.4:** a per-frame latency at the 95th percentile on the roadside unit, an accuracy floor, and at least three field conditions.

#### [capstone#what-is-submitted] Six things the assessor looks for in the public repository

*The repository*

- **Data card and split report.** What the data is, the guideline, the agreement score, no site shared across splits.
- **Field-conditions test set.** Paired with the clean test set, and how it was made.
- **Benchmark table.** Latency, size and accuracy per operating point, the machine named, the chosen row marked.
- **Model card.** Results per condition and district with intervals, the privacy controls and their measured recall, the monitors.
- **Release memo.** One page: release, release with a human check, or hold, per condition, against a rule stated in advance.
- **The trail.** Training logs, checkpoints, commit history, and one command that re-runs the evaluation.

#### [capstone#rubric] The rubric

*How the repository is scored*

| Item | Value |
| --- | --- |
| Data and splits | 20% |
| Model quality and error analysis | 25% |
| Latency budget | 20% |
| Field-condition tests and privacy | 25% |
| Reproducibility | 10% |

> **A claim without a number is a plan, not a result.** And a number without its condition, its count and its interval is half a result.

#### [capstone#format] Eight minutes of talk, then four minutes on a failure the assessor picks

*Format*

- **1 · The budget and the result.** The task, the budget you stated, the operating point you chose. Two minutes.
- **2 · The evidence.** The table by condition with intervals, the error breakdown, the release decision. Four minutes.
- **3 · What you would not release.** The weakest condition, what happens under it, the next step. Two minutes.

> **Then the failure case.** The assessor shows one frame or clip your service gets wrong, and names a held-back condition. You re-run the evaluation on it and explain what you see.

#### [capstone#running-order] Who presents when

*Running order*

| # | Presenter | Service |
| --- | --- | --- |
| 1 |  |  |
| 2 |  |  |
| 3 |  |  |
| 4 |  |  |
| 5 |  |  |
| 6 |  |  |
| 7 |  |  |
| 8 |  |  |

#### [capstone#final-prep] Final prep · 20 minutes

*Before the first defence*

Clone your repository into a new folder and run the evaluation command once.

#### [capstone#now-presenting] Talk

*Now presenting*

##### Failure case

#### [capstone#review] What the defences showed

*Trainer review*

- **Strongest evidence.** Tables and decisions that several projects got right.
- **Common gaps.** What most repositories would fix next.
- **One surprise.** A failure nobody expected, and what it taught.

#### [capstone#what-next] Keep going

*After the course*

- **The follow-up task:** run your service on one condition you did not test this week, and add the row to your model card
- **Your repository stays yours:** keep the evaluation command working as the model changes
- **Next Academy course:** AE-408 Multimodal and Speech Systems Engineering, for vision-language models and multimodal applications

#### [capstone#thanks] Thank you

*SDAIA Academy*

Computer Vision Engineering
