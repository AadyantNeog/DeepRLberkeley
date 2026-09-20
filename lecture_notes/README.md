# CS 185/285 Deep Reinforcement Learning - Lecture Notes

This folder contains the completed notes for all 25 supplied lectures: 146,607 transcript lines reconciled with 844 slide pages.

For topic-by-topic cross-references to Sutton and Barto's *Reinforcement Learning: An Introduction* (second edition), see the [course-to-book reference guide](COURSE_TO_RLBOOK_REFERENCE.md).

Every topic keeps its sources and interpretation separate:

1. **What the lecturer said - transcript only:** a concise, faithful paraphrase of the supplied transcript, including qualifications, examples, questions, and logistics where they affect the record.
2. **Source reconciliation:** notation recovered from slides, slide-only material, and disclosed transcript/slide mismatches.
3. **Additional explanation:** independent intuition, derivations, examples, and study guidance.

## Lecture manifest

| Lecture | Topic | Notes | Source status |
|---:|---|---|---|
| 01 | Introduction | [Lecture 01](lecture_01_introduction.md) | Supplied transcript ends mid-explanation; remaining slides isolated |
| 02 | Behavioral Cloning | [Lecture 02](lecture_02_behavioral_cloning.md) | Complete |
| 03 | Behavioral Cloning, Part 2 | [Lecture 03](lecture_03_behavioral_cloning_part_2.md) | Supplied transcript ends mid-recipe; remaining slides isolated |
| 04 | Reinforcement Learning Basics | [Lecture 04](lecture_04_rl_basics.md) | Complete supplied transcript; opening continues the prior lecture |
| 05 | Policy Gradients | [Lecture 05](lecture_05_policy_gradients.md) | Complete |
| 06 | Actor-Critic | [Lecture 06](lecture_06_actor_critic.md) | Complete |
| 07 | Value-Based Reinforcement Learning | [Lecture 07](lecture_07_value_based_rl.md) | Complete |
| 08 | Q-Learning in Practice | [Lecture 08](lecture_08_q_learning_in_practice.md) | Supplied transcript ends mid-question |
| 09 | Advanced Policy Gradients, Part 1 | [Lecture 09](lecture_09_advanced_policy_gradients_part_1.md) | Complete |
| 10 | Advanced Policy Gradients, Part 2 | [Lecture 10](lecture_10_advanced_policy_gradients_part_2.md) | Complete; spoken/slide notation discrepancies disclosed |
| 11 | Variational Inference | [Lecture 11](lecture_11_variational_inference.md) | Supplied transcript ends mid-teaser |
| 12 | Variational Inference in RL | [Lecture 12](lecture_12_variational_inference_in_rl.md) | Complete |
| 13 | Control as Variational Inference | [Lecture 13](lecture_13_control_as_inference.md) | Supplied transcript ends mid-Q&A; remaining slides isolated |
| 14 | RL with Sequences and LLMs | [Lecture 14](lecture_14_rl_with_sequences_and_llms.md) | Complete |
| 15 | Model-Based RL, Part 1 | [Lecture 15](lecture_15_model_based_rl_part_1.md) | Supplied transcript ends mid-question |
| 16 | Model-Based RL, Part 2 | [Lecture 16](lecture_16_model_based_rl_part_2.md) | Complete |
| 17 | Offline Reinforcement Learning, Part 1 | [Lecture 17](lecture_17_offline_rl_part_1.md) | Supplied transcript ends mid-sentence; remaining slides isolated |
| 18 | Offline RL, Part 2 | [Lecture 18](lecture_18_offline_rl_part_2.md) | Complete; explicitly deferred slides isolated |
| 19 | Exploration | [Lecture 19](lecture_19_exploration.md) | Complete; opening finishes deferred Lecture 18 material |
| 20 | Reinforcement Learning Theory | [Lecture 20](lecture_20_rl_theory.md) | Supplied transcript ends mid-question |
| 21 | Midterm Review, Part 1 | [Lecture 21](lecture_21_midterm_review_part_1.md) | Complete session; unspoken continuation slides isolated |
| 22 | Midterm Review, Part 2 | [Lecture 22](lecture_22_midterm_review_part_2.md) | Complete; cross-deck continuation disclosed |
| 23 | Exploration and Skill Learning | [Lecture 23](lecture_23_advanced_exploration.md) | Complete |
| 24 | Multi-Task and Hierarchical RL | [Lecture 24](lecture_24_multi_task_and_hierarchical_rl.md) | Technical lecture complete; closing transcript truncated and unspoken slides isolated |
| 25 | Challenges and Open Problems | [Lecture 25](lecture_25_challenges_and_open_problems.md) | Complete |

## Reading and source conventions

- Each numbered topic has an inclusive transcript line range. Across a lecture, these ranges form one monotonic, gap-free partition of every supplied line.
- Transcript-only prose does not silently import slide text or outside knowledge.
- Information visible only in the slide deck appears under **Source reconciliation** or in a clearly labeled slide-only appendix.
- A source cutoff is marked explicitly; missing speech is never reconstructed.
- Inline mathematics uses `$...$` and display mathematics uses `$$...$$` so symbols render consistently in Markdown math viewers.
- Every note ends with consolidated takeaways, key equations, a glossary, self-check questions, and a source coverage checklist.

## Collection QA

- [x] Exactly 25 lecture-note files are present and linked above.
- [x] All transcript and slide paths resolve to the supplied local sources.
- [x] Metadata line counts and PDF page counts match the source files.
- [x] Every transcript line is assigned exactly once in increasing, non-overlapping topic ranges.
- [x] All 844 PDF pages were rendered and visually inspected.
- [x] Transcript, slide-only, and independent explanation layers remain visibly separated.
- [x] Markdown structure, math delimiters, control characters, placeholders, trailing whitespace, and final newlines were checked collection-wide.
