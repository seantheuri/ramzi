import argparse
import json
import random
import subprocess
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List
import tempfile
import copy

import pandas as pd

# --- Constants / Paths -------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
ANALYSIS_DIR = ROOT_DIR / "analysis"
RESULTS_DIR = ROOT_DIR / "results"

# Ensure results directory exists
RESULTS_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# Helper: placeholder analysis function (to be implemented in Experiment 4)
# ----------------------------------------------------------------------------

def analyze_script(script_path: Path, analysis_a_path: Path, analysis_b_path: Path) -> Dict:
    """Extract objective metrics & human-likeness features from a mix script.

    Parameters
    ----------
    script_path : Path
        Path to generated mix-script JSON.
    analysis_a_path : Path
        Out-going track (Deck A) analysis JSON.
    analysis_b_path : Path
        Incoming track (Deck B) analysis JSON.

    Returns
    -------
    Dict[str, Any]
        Flat dictionary with calculated metrics.
    """

    # -------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------
    try:
        script_data = json.loads(script_path.read_text())
    except Exception:
        return {"analysis_error": "failed_to_load_script"}

    try:
        analysis_a = json.loads(analysis_a_path.read_text())
        analysis_b = json.loads(analysis_b_path.read_text())
    except Exception:
        return {"analysis_error": "failed_to_load_analysis"}

    commands = script_data.get("script", [])

    # Helper to fetch BPM
    def get_bpm(analysis: Dict) -> float:
        try:
            return float(analysis.get("bpm", 0))
        except (TypeError, ValueError):
            return 0.0

    bpm_a = get_bpm(analysis_a) or 1.0  # avoid div-by-zero

    # -------------------------------------------------------------
    # 1. Transition length in beats
    # -------------------------------------------------------------
    transition_start_time = None  # play Deck B
    for cmd in commands:
        if cmd.get("command") == "play" and cmd.get("params", {}).get("deck") == "B":
            transition_start_time = cmd.get("time")
            break

    transition_end_time_candidates = []
    # stop deck A
    for cmd in commands:
        if cmd.get("command") == "stop" and cmd.get("params", {}).get("deck") == "A":
            transition_end_time_candidates.append(cmd.get("time"))
    # channel fader fade to zero on deck A
    for cmd in commands:
        if (
            cmd.get("command") == "set_parameter"
            and cmd.get("params", {}).get("deck") == "A"
            and cmd.get("params", {}).get("parameter") == "channel_fader"
        ):
            p = cmd["params"]
            # if value is 0 or near-zero assume mix-out
            if p.get("value", 1.0) <= 0.05:
                fade_end = cmd.get("time", 0) + float(p.get("fade_duration", 0))
                transition_end_time_candidates.append(fade_end)

    transition_end_time = max(transition_end_time_candidates) if transition_end_time_candidates else None

    if transition_start_time is not None and transition_end_time is not None and transition_end_time > transition_start_time:
        transition_duration_sec = transition_end_time - transition_start_time
        transition_length_beats = transition_duration_sec / (60.0 / bpm_a)
    else:
        transition_length_beats = None

    # -------------------------------------------------------------
    # 2. Mix-out and mix-in structural labels
    # -------------------------------------------------------------
    def find_label(structure: List[Dict], timestamp: float) -> str:
        for seg in structure:
            if seg.get("start", 0) <= timestamp < seg.get("end", 0):
                return seg.get("label", "unknown")
        return "unknown"

    mix_out_label = None
    if transition_start_time is not None:
        mix_out_label = find_label(analysis_a.get("structural_analysis", []), transition_start_time)

    # For mix-in label, determine cue time for deck B (seek_to_time before play)
    cue_time_b = 0.0
    for cmd in commands:
        if cmd.get("command") == "seek_to_time" and cmd.get("params", {}).get("deck") == "B":
            cue_time_b = cmd.get("params", {}).get("time_in_seconds", 0.0)
            # Assume last such command before play overrides
        if cmd.get("command") == "play" and cmd.get("params", {}).get("deck") == "B":
            break
    mix_in_label = find_label(analysis_b.get("structural_analysis", []), cue_time_b)

    # -------------------------------------------------------------
    # 3. Harmonic score (Camelot wheel)
    # -------------------------------------------------------------
    def parse_camelot(camelot: str):
        if not camelot:
            return None, None
        try:
            number = int(camelot[:-1])
            letter = camelot[-1].upper()
            return number, letter
        except Exception:
            return None, None

    num_a, let_a = parse_camelot(analysis_a.get("key_camelot", ""))
    num_b, let_b = parse_camelot(analysis_b.get("key_camelot", ""))

    harmonic_score = 0
    if None not in (num_a, num_b):
        if num_a == num_b and let_a == let_b:
            harmonic_score = 2
        elif (abs(num_a - num_b) == 1 and let_a == let_b) or (num_a == num_b and let_a != let_b):
            harmonic_score = 1

    # -------------------------------------------------------------
    # 4. Tempo / key changes usage
    # -------------------------------------------------------------
    key_shift_count = sum(1 for cmd in commands if cmd.get("command") == "key_shift")

    bpm_change_a_percent = 0.0
    bpm_change_b_percent = 0.0
    for cmd in commands:
        if cmd.get("command") == "load_track":
            deck = cmd.get("params", {}).get("deck")
            tgt_bpm = cmd.get("params", {}).get("target_bpm")
            if tgt_bpm is None:
                continue
            try:
                tgt_bpm_f = float(tgt_bpm)
            except (TypeError, ValueError):
                continue
            if deck == "A":
                orig = bpm_a
                bpm_change_a_percent = abs(tgt_bpm_f - orig) / orig * 100.0
            elif deck == "B":
                orig_b = get_bpm(analysis_b)
                bpm_change_b_percent = abs(tgt_bpm_f - orig_b) / orig_b * 100.0

    # -------------------------------------------------------------
    return {
        "transition_length_beats": round(transition_length_beats, 2) if transition_length_beats is not None else None,
        "mix_out_label": mix_out_label,
        "mix_in_label": mix_in_label,
        "harmonic_score": harmonic_score,
        "key_shift_count": key_shift_count,
        "bpm_change_a_percent": round(bpm_change_a_percent, 2),
        "bpm_change_b_percent": round(bpm_change_b_percent, 2),
    }

# Attempt to import validation util from src.llm (optional)
try:
    from src.llm import validate_mix_script  # type: ignore
except Exception:
    def validate_mix_script(_data):  # type: ignore
        return False

# ----------------------------------------------------------------------------
# Experiment 1 – Model vs Prompting-style comparison
# ----------------------------------------------------------------------------

def run_experiment_1(n_pairs: int = 10):
    """Experiment 1: Evaluate how model and prompting style influence output.

    The experiment compares gpt-4o against itself using two different
    prompting strategies (direct vs chain-of-thought).
    """

    exp_name = "exp1_model_vs_prompting"
    exp_dir = RESULTS_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Models under test – mapping: nickname → OpenAI model string
    models_to_test: Dict[str, str] = {
        "4o": "gpt-4o",
    }

    prompting_styles: List[str] = ["direct", "cot"]

    # ---------------------------------------------------------------------
    # Build track pairs – choose n_pairs unique pairs from analysis dir
    # ---------------------------------------------------------------------
    analysis_files = sorted(ANALYSIS_DIR.glob("*_analysis.json"))
    if len(analysis_files) < 2:
        print("[Experiment 1] Not enough analysis files in the analysis/ directory.")
        return

    all_pairs = list(combinations(analysis_files, 2))
    if len(all_pairs) > n_pairs:
        random.seed(42)
        selected_pairs = random.sample(all_pairs, n_pairs)
    else:
        selected_pairs = all_pairs

    print(f"[Experiment 1] Running on {len(selected_pairs)} track pairs × "
          f"{len(models_to_test)} models × {len(prompting_styles)} styles")

    # Collect results for DataFrame
    results_records: List[Dict] = []

    # ---------------------------------------------------------------------
    for analysis_a, analysis_b in selected_pairs:
        track_a_name = analysis_a.stem.replace("_analysis", "")
        track_b_name = analysis_b.stem.replace("_analysis", "")
        track_pair_label = f"{track_a_name}_vs_{track_b_name}"

        for model_nick, model_id in models_to_test.items():
            for style in prompting_styles:
                prompt_base = (
                    "Create a professional mix that showcases both tracks effectively."
                )
                if style == "cot":
                    prompt = (
                        f"{prompt_base} "
                        "First, think step-by-step about your mixing strategy. "
                        "Explain your choice of mix-in/out points and techniques in a <thinking> block. "
                        "After the thinking block, generate the final JSON output."
                    )
                else:
                    prompt = prompt_base

                # Output path for generated script
                output_filename = (
                    f"{model_id}_{style}_{track_pair_label}.json".replace(" ", "_")
                )
                script_output_path = exp_dir / output_filename

                # Construct command to call llm.py
                cmd = [
                    sys.executable,
                    str(SRC_DIR / "llm.py"),
                    str(analysis_a),
                    str(analysis_b),
                    "-o",
                    str(script_output_path),
                    "-p",
                    prompt,
                    "-m",
                    model_id,
                ]

                print("\n------------------------------------------------------------")
                print("[Experiment 1] Running: " + " ".join(cmd))

                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                )

                if completed.returncode != 0:
                    print(f"[Experiment 1] llm.py failed: {completed.stderr}")
                    is_valid = False
                    script_data = {}
                else:
                    try:
                        with script_output_path.open() as f:
                            script_data = json.load(f)
                        is_valid = validate_mix_script(script_data)
                    except Exception as e:
                        print(f"[Experiment 1] Failed to read/validate script: {e}")
                        is_valid = False
                        script_data = {}

                # Analyse script (objective metrics placeholder)
                metrics = {}
                if script_output_path.exists():
                    metrics = analyze_script(
                        script_output_path, analysis_a, analysis_b
                    )

                # Derived descriptive stats
                command_count = len(script_data.get("script", [])) if script_data else 0
                technique_highlights = ",".join(
                    script_data.get("technique_highlights", [])
                ) if script_data else ""

                record: Dict = {
                    "experiment_name": exp_name,
                    "model_name": model_id,
                    "prompting_style": style,
                    "track_pair": track_pair_label,
                    "is_valid": is_valid,
                    "command_count": command_count,
                    "technique_highlights": technique_highlights,
                }
                # Merge metrics
                record.update(metrics)

                results_records.append(record)

    # ---------------------------------------------------------------------
    # Save aggregated results to CSV
    # ---------------------------------------------------------------------
    df = pd.DataFrame(results_records)
    csv_path = RESULTS_DIR / "experimental_results.csv"

    if csv_path.exists():
        df.to_csv(csv_path, mode="a", index=False, header=False)
    else:
        df.to_csv(csv_path, index=False)

    print(f"\n[Experiment 1] Results written to {csv_path} (appended)")

# ----------------------------------------------------------------------------
# Experiment 2 – Controllability via Prompt Engineering
# ----------------------------------------------------------------------------

def run_experiment_2(n_pairs: int = 10):
    """Experiment 2: Assess whether specific creative instructions are obeyed."""

    exp_name = "exp2_prompt_control"
    exp_dir = RESULTS_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Build list of track pairs
    analysis_files = sorted(ANALYSIS_DIR.glob("*_analysis.json"))
    if len(analysis_files) < 2:
        print("[Experiment 2] Not enough analysis files found for the experiment.")
        return
    all_pairs = list(combinations(analysis_files, 2))
    if len(all_pairs) > n_pairs:
        random.seed(42)
        selected_pairs = random.sample(all_pairs, n_pairs)
    else:
        selected_pairs = all_pairs

    print(f"[Experiment 2] Running on {len(selected_pairs)} track pairs...")

    # Prompts to evaluate, from vague to specific
    prompts_to_test: Dict[str, str] = {
        "vibe_chill": (
            "Create a relaxed, chilled-out vibe. The transition should feel "
            "seamless and natural, letting the music breathe."
        ),
        "style_berlin_techno": (
            "Mix these tracks like a classic Berlin techno DJ. Think long, "
            "filter-heavy blends and hypnotic loops."
        ),
        "technique_acapella_intro": (
            "Isolate the vocals (acapella) of the incoming track and layer them "
            "over the beat of the outgoing track for 8 beats before the full "
            "song comes in."
        ),
        "technical_hard_swap": (
            "Perform a hard swap. At the end of a phrase in Track A, cut the "
            "bass on A, then immediately start Track B with its bass cut. After "
            "4 beats, swap the basslines instantly: bring in B's bass and kill A's."
        ),
        "technical_loop_roll": (
            "On the last 4 beats of track A, apply a beat repeat effect, starting "
            "with a 1-beat loop and halving it every beat down to 1/8th. At the "
            "same time, apply a high-pass filter sweep. Drop track B on the first "
            "beat of the next bar."
        ),
    }

    model_id = "gpt-4o"  # Best available model

    results_records: List[Dict] = []

    for analysis_a, analysis_b in selected_pairs:
        track_pair_label = f"{analysis_a.stem.replace('_analysis','')}_vs_{analysis_b.stem.replace('_analysis','')}"

        for prompt_id, prompt_text in prompts_to_test.items():
            output_path = exp_dir / f"{prompt_id}_{track_pair_label}.json"

            cmd = [
                sys.executable,
                str(SRC_DIR / "llm.py"),
                str(analysis_a),
                str(analysis_b),
                "-o",
                str(output_path),
                "-p",
                prompt_text,
                "-m",
                model_id,
            ]

            print("\n------------------------------------------------------------")
            print(f"[Experiment 2] Prompt '{prompt_id}' on pair '{track_pair_label}' → running llm.py ...")

            completed = subprocess.run(cmd, capture_output=True, text=True)
            if completed.returncode != 0:
                print(f"[Experiment 2] llm.py error: {completed.stderr}")
                is_valid = False
                script_data = {}
            else:
                try:
                    with output_path.open() as f:
                        script_data = json.load(f)
                    is_valid = validate_mix_script(script_data)
                except Exception as e:
                    print(f"[Experiment 2] Failed to read/validate script: {e}")
                    is_valid = False
                    script_data = {}

            metrics = {}
            if output_path.exists():
                metrics = analyze_script(output_path, analysis_a, analysis_b)

            record: Dict = {
                "experiment_name": exp_name,
                "prompt_id": prompt_id,
                "prompt_text": prompt_text,
                "generated_description": script_data.get("description", ""),
                "track_pair": track_pair_label,
                "model_name": model_id,
                "prompting_style": "direct",
                "is_valid": is_valid,
                "command_count": len(script_data.get("script", [])) if script_data else 0,
                "technique_highlights": ",".join(script_data.get("technique_highlights", [])) if script_data else "",
            }
            record.update(metrics)
            results_records.append(record)

    # Append to CSV
    df = pd.DataFrame(results_records)
    csv_path = RESULTS_DIR / "experimental_results.csv"
    if csv_path.exists():
        df.to_csv(csv_path, mode="a", index=False, header=False)
    else:
        df.to_csv(csv_path, index=False)

    print(f"\n[Experiment 2] Results appended to {csv_path}")

# ----------------------------------------------------------------------------
# Experiment 3 – Ablation Study on Input Data
# ----------------------------------------------------------------------------

def run_experiment_3(n_pairs: int = 15):
    """Experiment 3: Ablation study – evaluate multiple track pairs.

    Parameters
    ----------
    n_pairs : int
        Number of unique track pairs to test. Default 5.
    """

    exp_name = "exp3_ablation"
    exp_dir = RESULTS_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Build list of track pairs similar to experiment 1
    analysis_files = sorted(ANALYSIS_DIR.glob("*_analysis.json"))
    if len(analysis_files) < 2:
        print("[Experiment 3] Not enough analysis files.")
        return
    all_pairs = list(combinations(analysis_files, 2))
    if len(all_pairs) > n_pairs:
        random.seed(123)
        selected_pairs = random.sample(all_pairs, n_pairs)
    else:
        selected_pairs = all_pairs

    print(f"[Experiment 3] Running on {len(selected_pairs)} track pairs × {len(['full_data','no_structure','no_lyrics'])} ablations")

    ablation_conditions = ["full_data", "no_structure", "no_lyrics"]
    model_id = "gpt-4o"

    results_records: List[Dict] = []

    # Temporary workspace for modified files
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        for analysis_a_path, analysis_b_path in selected_pairs:
            # Load originals
            with analysis_a_path.open() as f:
                original_a = json.load(f)
            with analysis_b_path.open() as f:
                original_b = json.load(f)

            track_pair_label = f"{analysis_a_path.stem.replace('_analysis','')}_vs_{analysis_b_path.stem.replace('_analysis','')}"

            for condition in ablation_conditions:
                # Deep copies of data
                data_a = copy.deepcopy(original_a)
                data_b = copy.deepcopy(original_b)

                if condition == "no_structure":
                    data_a.pop("structural_analysis", None)
                    data_a.pop("sections", None)
                    data_b.pop("structural_analysis", None)
                    data_b.pop("sections", None)
                elif condition == "no_lyrics":
                    data_a["lyrics_text"] = ""
                    data_a["lyrics_timed"] = []
                    data_b["lyrics_text"] = ""
                    data_b["lyrics_timed"] = []

                # Write to temp files unique per iteration
                mod_a_path = tmpdir / f"a_{track_pair_label}_{condition}.json"
                mod_b_path = tmpdir / f"b_{track_pair_label}_{condition}.json"
                mod_a_path.write_text(json.dumps(data_a, indent=2))
                mod_b_path.write_text(json.dumps(data_b, indent=2))

                # Output script path
                script_output = exp_dir / f"{condition}_{track_pair_label}.json"

                # Call llm.py
                cmd = [
                    sys.executable,
                    str(SRC_DIR / "llm.py"),
                    str(mod_a_path),
                    str(mod_b_path),
                    "-o",
                    str(script_output),
                    "-m",
                    model_id,
                ]

                print("\n------------------------------------------------------------")
                print(f"[Experiment 3] Condition '{condition}' → executing llm.py ...")

                completed = subprocess.run(cmd, capture_output=True, text=True)
                if completed.returncode != 0:
                    print(f"[Experiment 3] llm.py error: {completed.stderr}")
                    is_valid = False
                    script_data = {}
                else:
                    try:
                        with script_output.open() as f:
                            script_data = json.load(f)
                        is_valid = validate_mix_script(script_data)
                    except Exception as e:
                        print(f"[Experiment 3] Failed to load/validate script: {e}")
                        is_valid = False
                        script_data = {}

                metrics = {}
                if script_output.exists():
                    metrics = analyze_script(script_output, analysis_a_path, analysis_b_path)

                record: Dict = {
                    "experiment_name": exp_name,
                    "ablation_condition": condition,
                    "track_pair": track_pair_label,
                    "model_name": model_id,
                    "is_valid": is_valid,
                    "command_count": len(script_data.get("script", [])) if script_data else 0,
                    "technique_highlights": ",".join(script_data.get("technique_highlights", [])) if script_data else "",
                }
                record.update(metrics)
                results_records.append(record)

    # Save to CSV
    df = pd.DataFrame(results_records)
    csv_path = RESULTS_DIR / "experimental_results2.csv"
    if csv_path.exists():
        df.to_csv(csv_path, mode="a", index=False, header=False)
    else:
        df.to_csv(csv_path, index=False)

    print(f"\n[Experiment 3] Results appended to {csv_path}")

# ----------------------------------------------------------------------------
# CLI Entry-point
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RAMZI – Experimental Evaluation Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-e",
        "--experiment",
        choices=["1", "2", "3", "all"],
        default="1",
        help="Which experiment to run",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=10,
        help="Number of track pairs to sample for experiments 1, 2, and 3",
    )

    args = parser.parse_args()

    if args.experiment in ("1", "all"):
        run_experiment_1(n_pairs=args.n_pairs)

    if args.experiment in ("2", "all"):
        run_experiment_2(n_pairs=args.n_pairs)

    if args.experiment in ("3", "all"):
        run_experiment_3(n_pairs=args.n_pairs)

    # Placeholder for future experiments 4
    if args.experiment in ("4", "all"):
        print("[TODO] Experiment 4 not implemented yet.")


if __name__ == "__main__":
    main() 