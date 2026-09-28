"""Q8 - install and check in volis: score both translators the way volis
runs them, and keep the new one only if it's better.

1. Before swapping: translate the test set through volis with its current
   .gguf (volis --translate, M7.8), and score chrF per direction. This is the
   fair baseline: same program, same compression, same decoding.
2. Move volis's current .gguf into models/mt-parked/ (never deleted), and copy
   the new one into models/mt/ (volis refuses to run with two).
3. volis --report must list the new file; translate the test set again.
4. Pass if the new model beats step 1 in every direction, lost no more than 3
   chrF points against Q5's full-precision score, and isn't refused more often.
   If it doesn't pass, the old translator is put back automatically.
5. Print both scores and how to swap by hand.

Text is fed to volis from Python as UTF-8 bytes, never through a PowerShell
pipe (Windows PowerShell 5.1 would break accents and Arabic).
"""

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import Stop, heading, run_dir, run_stage, stage_result  # noqa: E402
from qwen_common import chrf_by_direction, read_jsonl, volis_dir, volis_exe  # noqa: E402
from whisper_common import write_json  # noqa: E402

LOSS_LIMIT = 3.0


def translate_through_volis(test: list) -> tuple:
    """(translations, refused count, refusal reasons) for the test set, one
    volis run per direction. Reasons are volis's stderr lines "line N: ..."."""
    out = [""] * len(test)
    refused = 0
    reasons = [""] * len(test)
    directions = sorted({(p["source_lang"], p["target_lang"]) for p in test})
    for source, target in directions:
        idx = [i for i, p in enumerate(test) if (p["source_lang"], p["target_lang"]) == (source, target)]
        text = "\n".join(test[i]["source"].replace("\n", " ") for i in idx) + "\n"
        done = subprocess.run([str(volis_exe()), "--translate", source, target],
                              input=text.encode("utf-8"), capture_output=True)
        if done.returncode != 0:
            raise Stop(f"volis --translate {source} {target} failed: "
                       f"{done.stderr.decode('utf-8', 'replace').strip()[-800:]}")
        lines = done.stdout.decode("utf-8").split("\n")[:len(idx)]
        if len(lines) != len(idx):
            raise Stop(f"volis returned {len(lines)} lines for {len(idx)} ({source}>{target})")
        for i, line in zip(idx, lines):
            out[i] = line.strip()
            refused += not line.strip()
        for err in done.stderr.decode("utf-8", "replace").splitlines():
            if err.startswith("line ") and ":" in err:
                number = err[5:err.index(":")]
                if number.isdigit() and 0 < int(number) <= len(idx):
                    reasons[idx[int(number) - 1]] = err.split(":", 1)[1].strip()
    return out, refused, reasons


def show(label: str, scores: dict, refused: int) -> None:
    for d, s in scores.items():
        print(f"  {label:8} {d:10} chrF {s['chrF']:6.2f}")
    print(f"  {label:8} refused by volis's guards: {refused}")


def work(job: dict) -> dict:
    new_file = Path(stage_result(job, "Q7")["gguf"])
    full_precision = {k: v for k, v in stage_result(job, "Q5").items() if isinstance(v, dict)}
    test = read_jsonl(run_dir(job) / "data" / "test.jsonl")
    mt = volis_dir() / "models" / "mt"
    parked = volis_dir() / "models" / "mt-parked"
    current = sorted(mt.glob("*.gguf"))
    if len(current) != 1:
        raise Stop(f"{mt} should hold exactly one .gguf, and holds {[p.name for p in current]}")
    old_file = current[0]
    if old_file.name == new_file.name:
        raise Stop(f"volis already runs a file called {new_file.name}; give the new one another "
                   "output.file_name, so the two can't be confused")

    heading(f"1. volis's current translator ({old_file.name}) on the test set")
    old_hyps, old_refused, old_reasons = translate_through_volis(test)
    old = chrf_by_direction(test, old_hyps)
    show("current", old, old_refused)

    heading(f"2. Installing {new_file.name}")
    parked.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old_file), parked / old_file.name)
    shutil.copy2(new_file, mt / new_file.name)
    print(f"  parked {old_file.name} in {parked}; installed {new_file.name} in {mt}")

    def restore(reason: str) -> None:
        (mt / new_file.name).unlink(missing_ok=True)
        shutil.move(str(parked / old_file.name), mt / old_file.name)
        raise Stop(f"{reason}\n  The previous translator ({old_file.name}) has been put back.")

    try:
        report = subprocess.run([str(volis_exe()), "--report"], capture_output=True)
        if new_file.name not in report.stdout.decode("utf-8", "replace"):
            restore(f"volis --report doesn't list {new_file.name}")
        heading(f"3. The new translator on the test set, through volis")
        new_hyps, new_refused, new_reasons = translate_through_volis(test)
    except Stop:
        raise
    except Exception as e:  # noqa: BLE001 - anything here must put the old file back
        restore(f"checking the new translator failed: {e}")
    new = chrf_by_direction(test, new_hyps)
    show("new", new, new_refused)
    write_json(run_dir(job) / "scores" / "volis-test.json", {
        "current": {"file": old_file.name, "scores": old, "refused": old_refused},
        "new": {"file": new_file.name, "scores": new, "refused": new_refused},
        "translations": [{"direction": f"{p['source_lang']}>{p['target_lang']}", "source": p["source"],
                          "reference": p["target"], "current": o, "new": n,
                          "current_refused": ro, "new_refused": rn}
                         for p, o, n, ro, rn in zip(test, old_hyps, new_hyps, old_reasons, new_reasons)]})

    heading("4. Verdict")
    problems = []
    for d in new:
        print(f"  {d:10} current {old[d]['chrF']:6.2f} -> new {new[d]['chrF']:6.2f} "
              f"({new[d]['chrF'] - old[d]['chrF']:+.2f}); full precision {full_precision[d]['chrF']:.2f}")
        if new[d]["chrF"] <= old[d]["chrF"]:
            problems.append(f"{d} didn't beat the current translator")
        if full_precision[d]["chrF"] - new[d]["chrF"] > LOSS_LIMIT:
            problems.append(f"{d} lost more than {LOSS_LIMIT} chrF in conversion")
    if new_refused > old_refused:
        problems.append(f"volis refused {new_refused} of its translations, against {old_refused}")
    if problems:
        restore("the new translator didn't pass: " + "; ".join(problems))
    print(f"\n  The new translator is installed. To go back to the previous one:\n"
          f"    move {mt / new_file.name} somewhere else, then move\n"
          f"    {parked / old_file.name} into {mt}")
    return {"current": old, "new": new, "installed": new_file.name, "parked": old_file.name}


if __name__ == "__main__":
    run_stage("Q8", "install and check in volis", work)
