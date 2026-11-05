"""Load Experiment 6 ("rooms", 2AFC) behavioural data into tidy dataframes.

Each of the 72 choices runs:

    choice_intro -> observe x N (automatic presses, both rooms) -> room_choice
                 -> gold_pause -> gold (button choice in the chosen room)

Typical use:

    from load_data_rooms import load_participant, load_directory

    d = load_directory("expt/Experiment6 - rooms/data/3a4k_gold")
    df_choice, df_coin = d["choice"], d["coin"]   # join on [subject_id, choice]

Helpers are reused from load_data.py (Experiment 4) wherever the trial fields match.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from emp_utils import canonical_count_matrix, array_to_hist
from load_data import (
    _read_trials, _flatten, _by_task, _session as _session_exp4,
    _attention_df, _feedback_df, _coin_df as _coin_df_exp4, dirichlet_entropy,
    bonus_df,
)

PHASES = ("meta", "observe", "choice", "coin", "attention", "feedback")

BUTTONS = ("blue", "red", "green")              # row order of counts arrays, as in load_data
OUTCOMES = ("up", "down", "left", "right")      # column order of counts arrays, as in load_data
BUTTON_MAP = {'blue': 0, 'red': 1, 'green': 2}
OUTCOME_MAP = {'up': 0, 'left': 1, 'down': 2, 'right': 3}


#----------------------------------------------------------------------------#
# Helpers
#----------------------------------------------------------------------------#

def _session(trials, path):
    """Session fields from load_data, plus the ones new to this experiment."""
    session = _session_exp4(trials, path)
    for k in ("objective", "confidence_on"):
        session[k] = trials[-1].get(k)
    return session


def _counts_array(counts):
    """{button: {outcome: n}} -> n_arms x 4 array (rows blue, red, green; cols up, down, left, right)."""
    return np.array([[counts[b][o] for o in OUTCOMES] for b in BUTTONS if b in counts])


def _history_str(C):
    """Canonical history string of a count matrix, matching the model's state keys."""
    n_arms, n_outcomes = C.shape
    return array_to_hist(canonical_count_matrix(C)[0], n_arms, n_outcomes)[1]


# per-room statistics added to the choice table, as `{stat}_{option}`, plus
# chosen_/unchosen_/_diff versions
ROOM_STATS = (
    "n_presses", "n_buttons_tried", "max_counts", "max_counts_fraction",
    "n_diff_outcomes", "total_entropy",
) + tuple(f"{s}_{b}" for s in ("counts", "n_diff_outcomes", "entropy") for b in BUTTONS)


def _room_summary(C, alpha, j):
    """Sample-count, outcome-diversity and Dirichlet-entropy summaries of one room.

    Mirrors the per-button columns of load_data._sample_df, computed from the
    room's tally at choice time. `n_diff_outcomes` (no button suffix) counts the
    distinct locations reached by any button in the room.
    """
    per_button = C.sum(axis=1)
    out = {
        f"n_buttons_tried_{j}": int((per_button > 0).sum()),
        f"max_counts_{j}": int(per_button.max()),
        f"max_counts_fraction_{j}": per_button.max() / per_button.sum() if per_button.sum() else np.nan,
        f"n_diff_outcomes_{j}": int((C.sum(axis=0) > 0).sum()),
    }
    entropies = []
    for i, b in enumerate(BUTTONS[:C.shape[0]]):
        out[f"counts_{b}_{j}"] = int(per_button[i])
        out[f"n_diff_outcomes_{b}_{j}"] = int((C[i] > 0).sum())
        out[f"entropy_{b}_{j}"] = dirichlet_entropy(C[i] + alpha)
        entropies.append(out[f"entropy_{b}_{j}"])
    out[f"total_entropy_{j}"] = sum(entropies)
    return out


def _preset_array(history, n_arms, n_outcomes):
    """Canonical preset history [[[a, o], n], ...] -> count matrix."""
    C = np.zeros((n_arms, n_outcomes), dtype=int)
    for (a, o), n in history:
        C[a, o] += n
    return C


#----------------------------------------------------------------------------#
# Task phases
#----------------------------------------------------------------------------#

def _meta_df(trials, session):
    """One row per participant: condition, totals, quiz and session duration."""
    elapsed = [t.get("time_elapsed") for t in trials if t.get("time_elapsed") is not None]
    attention = _by_task(trials, "attention_check")
    quiz = _by_task(trials, "comprehension")
    row = dict(session)
    row.update(
        n_trials=len(trials),
        duration_min=max(elapsed) / 60000 if elapsed else np.nan,
        n_choices_done=len(_by_task(trials, "room_choice")),
        n_attention=len(attention),
        attention_accuracy=attention[-1].get("attention_accuracy") if attention else np.nan,
    )
    for k in ("quiz_correct", "quiz_total", "quiz_passed", "quiz_attempt"):
        row[k] = quiz[-1].get(k) if quiz else np.nan
    return pd.DataFrame([row])


def _observe_df(trials, session):
    """One row per automatic press, in either room, before the room choice.

    `option` is the preset index (as in room_choice), `position` the screen slot.
    `counts_*`/`post_*` are the room's tally before the press, `counts_post_*` after.
    """
    rows = []
    for t in _by_task(trials, "observe"):
        row = {"subject_id": session["subject_id"]}
        row.update(
            choice=t.get("choice_num") - 1,
            option=t.get("option"),
            position=t.get("position"),
            obs=t.get("obs_num") - 1,
            button=BUTTON_MAP.get(t.get("observed_button")),
            outcome=OUTCOME_MAP.get(t.get("observed_outcome")),
        )
        row.update(_flatten(t.get("counts"), "counts"))
        row.update(_flatten(t.get("posterior_means"), "post"))
        row.update(_flatten(t.get("counts_post"), "counts_post"))
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df[["button", "outcome"]] = df[["button", "outcome"]].astype("Int64")
    return df


def _choice_df(trials, session):
    """One row per room choice, wide: one column block per option (preset index 0/1).

    `history_str_j` is the canonical history of option j's preset presses, in
    the same format as the model's `history_str`. `true_T_*` is the chosen
    room's generative T only -- the unchosen room's is never drawn.
    """
    rows = []
    for t in _by_task(trials, "room_choice"):
        option_order = t.get("option_order")
        chosen = t.get("chosen_option")
        row = {"subject_id": session["subject_id"], "alpha": session["alpha"]}
        row.update(
            choice=t.get("choice_num") - 1,
            preset_trial_index=t.get("preset_trial_index"),
            rt=t.get("rt"),
            chosen_option=chosen,
            chosen_position=t.get("chosen_position"),
            option_order=option_order,
        )
        for j, (counts, preset) in enumerate(zip(t.get("preset_counts"), t.get("preset_histories"))):
            C = _counts_array(counts)
            hist = _history_str(C)
            # the stored canonical preset must be the same history as the concrete tally
            preset_hist = _history_str(_preset_array(preset, *C.shape))
            if hist != preset_hist:
                raise ValueError(f"choice {row['choice']} option {j}: counts give {hist}, preset gives {preset_hist}")
            row[f"n_presses_{j}"] = t.get("n_preset_presses")[j]
            row[f"position_{j}"] = option_order.index(j)
            row[f"counts_array_{j}"] = C
            row[f"history_str_{j}"] = hist
            # the preset history in the encoding gen_rooms / fit_emp use: repr of (((a, o), count), ...)
            row[f"room_history_{j}"] = repr(tuple(((a, o), n) for (a, o), n in preset))
            row[f"t_{j}"] = int(C.sum())
            row.update(_room_summary(C, session["alpha"], j))
        row.update(_flatten(t.get("true_T"), "true_T"))
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    n_opt = len(t.get("preset_counts"))
    idx = np.arange(len(df))
    chosen = df.chosen_option.to_numpy()

    # what fit_emp's rooms likelihood reads: the chosen room's index k (as in
    # room_history_{k}) and the design constants
    df["action"] = df.chosen_option
    df["n_AFC"] = n_opt
    df["n_arms"], df["n_outcomes"] = df[f"counts_array_0"].iloc[0].shape
    df["expt"] = "rooms"

    def per_option(stat):
        return df[[f"{stat}_{j}" for j in range(n_opt)]].to_numpy()

    # chosen / unchosen copies of each per-room statistic, and their difference
    for stat in ("history_str",) + ROOM_STATS:
        vals = per_option(stat)
        df[f"chosen_{stat}"] = vals[idx, chosen]
        if n_opt == 2:
            df[f"unchosen_{stat}"] = vals[idx, 1 - chosen]
            if stat != "history_str":
                df[f"{stat}_diff"] = df[f"chosen_{stat}"] - df[f"unchosen_{stat}"]

    # did they pick the room with more / fewer presses? NaN when the rooms tie
    presses = per_option("n_presses")
    tie = (presses == presses.max(axis=1, keepdims=True)).all(axis=1)
    df["chose_more_presses"] = np.where(tie, np.nan, df.chosen_n_presses == presses.max(axis=1))
    df["chose_fewer_presses"] = np.where(tie, np.nan, df.chosen_n_presses == presses.min(axis=1))

    return df


def _coin_df(trials, session, df_choice):
    """One row per gold trial, via load_data._coin_df plus this design's extra fields.

    `room` from load_data is 1-based and equals choice_num, so it becomes the
    0-based `choice` key shared with the choice table.
    """
    df = _coin_df_exp4(trials, session)
    if df.empty:
        return df
    extra = pd.DataFrame([{
        "room": t.get("room_num"),
        "chosen_option": t.get("chosen_option"),
        "chosen_position": t.get("chosen_position"),
        "sampled_outcome": OUTCOME_MAP.get(t.get("sampled_outcome")),
        "collected": t.get("collected"),
        "collected_gold_total": t.get("collected_gold_total"),
    } for t in _by_task(trials, "gold")])
    df = df.drop(columns=["collected_gold"], errors="ignore").merge(extra, on="room", how="left")
    df["sampled_outcome"] = df["sampled_outcome"].astype("Int64")
    df["choice"] = df.pop("room") - 1
    if not df_choice.empty:
        df = df.merge(df_choice[["choice", "chosen_history_str"]], on="choice", how="left")
    return df


#----------------------------------------------------------------------------#
# Public API
#----------------------------------------------------------------------------#

def load_participant(path):
    """Load one participant's JSON file into a dataframe per task phase.

    Returns:
        dict of dataframes keyed by phase: "meta", "observe", "choice", "coin",
        "attention", "feedback". Phases not reached come back empty.
    """
    trials = _read_trials(path)
    session = _session(trials, path)
    if session["subject_id"] is None:
        session["subject_id"] = Path(path).stem
        print(f"subject_id missing from {path}, using source_file instead: {session['subject_id']}")

    df_choice = _choice_df(trials, session)
    data = {
        "meta": _meta_df(trials, session),
        "observe": _observe_df(trials, session),
        "choice": df_choice,
        "coin": _coin_df(trials, session, df_choice),
        "attention": _attention_df(trials, session),
        "feedback": _feedback_df(trials, session),
    }

    ## rename subject_id to pid in all
    for df in data.values():
        if "subject_id" in df.columns:
            df.rename(columns={"subject_id": "id"}, inplace=True)
        if "choice" in df.columns:
            df.rename(columns={"choice": "room"}, inplace=True)
    return data


def load_directory(data_dir, pattern="*.json"):
    """Load every participant file in a directory, concatenated per phase.

    Feedback is written to `data_dir/feedback.csv` and left out of the result,
    as in load_data.load_directory.
    """
    paths = [p for p in sorted(Path(data_dir).glob(pattern)) if p.name != 'pid_map.json']
    if not paths:
        raise FileNotFoundError(f"no files matching {pattern!r} in {data_dir}")

    loaded = []
    for p in paths:
        try:
            loaded.append(load_participant(p))
        except Exception as e:
            print(f"Error loading {p}: {e}")
    out = {}
    for phase in PHASES:
        parts = [d[phase] for d in loaded if not d[phase].empty]
        out[phase] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    out.pop('feedback').to_csv(Path(data_dir) / 'feedback.csv', index=False)
    out['bonuses'] = bonus_df(out['meta'], data_dir)
    return out
