# Extraction process notes (2026-07-17)

No hand edits were made to any inventory content — module_inventory.json is
verbatim the parsed reply of call_019_diagnosed_repair1.json. Full call chain:

1. calls 000-011: per-seed annotation (gemini-3.1-pro-preview, temp 0).
2. call 012 merge + 013-015 stock repair rounds: these three repair rounds
   chased a HARNESS BUG (LLM-returned provenance pids were JSON strings; the
   gate compared against ints, so every seed reconstructed with zero modules).
   Fixed in repo commit 62bf469; scripts below replayed the pipeline tail from
   the logged call_015 reply instead of re-running the annotate/merge calls.
3. script_replay_extraction.py: validation+coverage clean; gate failures for
   seeds 2, 5, 10, 11, 13 (real extraction distortions).
4. calls 016-018 (script_targeted_repair.py): repair rounds carrying the
   failing seeds' original sources; fixed 2, 5, 10.
5. call 019 (script_diagnosed_repair.py): one repair round with explicit
   line-by-line diagnoses for seeds 11 and 13 (stage-2 reward-dependent
   stickiness; both-stage outcome-dependent temperature; signed perseveration
   [-3,3]; updated-target stage-1 TD ordering). call_019's reply passed
   everything: 12/12 seeds within gate tolerance, 4 exact (delta 0.00).

reconstruction_report.json reflects the final call_019 inventory.
