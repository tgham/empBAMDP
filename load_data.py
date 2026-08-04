"""Load Experiment 4 behavioural data (jsPsych JSON) into tidy dataframes.

Typical use:

    from load_data import load_participant, load_directory

    d = load_participant("data/pilot/zdbc0hryabjsn3zikghb1hfx.json")
    df_sample, df_coin = d["sample"], d["coin"]
    df_ius, df_loc1, df_loc2 = d["IUS"], d["LOC1"], d["LOC2"]

    all_data = load_directory("data/pilot")   # every participant, concatenated
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gamma, digamma

from emp_utils import canonical_states, canonical_count_matrix, array_to_hist, canon_to_concrete


# jsPsych.data.addProperties() stamps these onto every trial retroactively, so
# they are session-level constants rather than per-trial measurements.
SESSION_COLS = [
    "subject_id", "study_id", "session_id",
    "belief_display", "alpha", "contextual", "alpha_ctx1", "alpha_ctx2",
    "button_upper", "button_lower", "n_rooms", "coins_collected", "bonus_gbp",
]

QUESTIONNAIRES = ("IUS", "LOC1", "LOC2")

PHASES = ("meta", "rooms", "sample", "coin", "attention") + QUESTIONNAIRES + ("feedback",)


#----------------------------------------------------------------------------#
# Helpers
#----------------------------------------------------------------------------#

def _read_trials(path):
    """Return the list of jsPsych trials from a saved data file.

    The backend stores jsPsych's already-serialised data, so the file is
    double-encoded: the outer json.load() yields a str, not a list.

    Args:
        path: path to a participant's .json file.

    Returns:
        list of dicts, one per jsPsych trial, in chronological order.
    """
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, str):
        raw = json.loads(raw)
    return raw


def _strip_html(s):
    """Reduce an HTML item/scale label to plain text."""
    if not isinstance(s, str):
        return s
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s)).strip()


def _flatten(nested, prefix):
    """Flatten {button: {outcome: value}} to {prefix_button_outcome: value}."""
    out = {}
    if not isinstance(nested, dict):
        return out
    for button, outcomes in nested.items():
        if isinstance(outcomes, dict):
            for outcome, value in outcomes.items():
                out[f"{prefix}_{button}_{outcome}"] = value
    return out


def _by_task(trials, task):
    return [t for t in trials if t.get("task") == task]


def _session(trials, path):
    """Session-level fields, identical on every trial."""
    session = {k: trials[-1].get(k) for k in SESSION_COLS}
    session["source_file"] = Path(path).name
    return session


def _gen_alpha(session, ctx):
    """The Dirichlet alpha a button's *true* transition function was drawn from.

    This is the generative prior only. The belief shown to the participant is
    always computed with the single session-level ALPHA regardless of context
    (see posteriorMean in js/render.js), so for a context-2 button the displayed
    posterior is deliberately misspecified w.r.t. this value.

    BUTTON_CTX is null for every button when the task runs non-contextually, in
    which case the single ALPHA is also the generative prior (see js/config.js).
    """
    if not session.get("contextual") or ctx is None or (isinstance(ctx, float) and np.isnan(ctx)):
        return session.get("alpha")
    return session.get("alpha_ctx1") if ctx == 1 else session.get("alpha_ctx2")


#----------------------------------------------------------------------------#
# Task phases
#----------------------------------------------------------------------------#

def _meta_df(trials, session):
    """One row per participant: condition, totals, and session duration."""
    elapsed = [t.get("time_elapsed") for t in trials if t.get("time_elapsed") is not None]
    attention = _by_task(trials, "attention_check")
    row = dict(session)
    row.update(
        n_trials=len(trials),
        duration_min=max(elapsed) / 60000 if elapsed else np.nan,
        n_rooms_done=len(_by_task(trials, "gold")),
        n_attention=len(attention),
        attention_accuracy=attention[-1].get("attention_accuracy") if attention else np.nan,
        completed_surveys=all(_by_task(trials, q) for q in QUESTIONNAIRES),
    )
    return pd.DataFrame([row])


def _rooms_df(trials, session):
    """One row per room: the true transition function and each button's prior."""
    rooms = {}
    for t in _by_task(trials, "room_intro"):
        r = rooms.setdefault(t.get("room_num"), {})
        r["intro_rt"] = t.get("rt")
        r.update(_flatten(t.get("true_T"), "trueT"))
        for button, ctx in (t.get("button_ctx") or {}).items():
            r[f"ctx_{button}"] = ctx
            r[f"gen_alpha_{button}"] = _gen_alpha(session, ctx)
    for t in _by_task(trials, "room_sampling"):
        r = rooms.setdefault(t.get("room_num"), {})
        r["practice"] = t.get("practice")
        r["n_presses"] = t.get("n_presses")

    rows = [{**session, "room_num": k, **v} for k, v in sorted(rooms.items())]
    return pd.DataFrame(rows)


def _sample_df(trials, session):
    """One row per button press during the sampling ("button testing") phase.

    `counts_*` and `post_*` are the belief state the press was made *from*;
    `counts_post_*` includes the outcome just observed.

    `post_*` is the belief the participant was shown, i.e. always
    (alpha + count) / (4*alpha + N) using the session-level `alpha` -- never the
    context-specific prior the true transitions were drawn from (see _gen_alpha).

    Ending sampling early (the tick button) is logged as its own row with
    ended_early=True and no chosen_button/outcome, so filter on
    `~df.ended_early` to get actual presses.
    """
    rows = []
    for t in trials:
        if t.get("task") not in ("sample", 
                                #  "practice_sample"
                                 ):
            continue
        row = dict(session)
        row.update(
            practice=t.get("task") == "practice_sample",
            room_num=t.get("room_num"),
            trial_num=t.get("trial_num"),
            chosen_button=t.get("chosen_button"),
            outcome=t.get("outcome"),
            rt=t.get("rt"),
            ended_early=t.get("ended_early"),
        )
        row.update(_flatten(t.get("counts"), "counts"))            # at decision time
        row.update(_flatten(t.get("posterior_means"), "post"))     # at decision time
        row.update(_flatten(t.get("counts_post"), "counts_post"))  # after the outcome
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["chosen_button_pos"] = np.where(
            df.chosen_button.isna(), None,
            np.where(df.chosen_button == df.button_upper, "upper", "lower"),
        )


    ### more formatting

    ## cols to remove
    rm_cols = ['study_id', 'session_id', 'belief_display', 
            #    'alpha', 
               'contextual','alpha_ctx1','alpha_ctx2','source_file','button_upper', 'button_lower', 'n_rooms', 'coins_collected', 
            #    'bonus_gbp',
            #    'chosen_button_pos'
               ]
    df.drop(columns=rm_cols, inplace=True, errors='ignore')

    ## ensure columns are ints
    int_cols = ['room_num', 'trial_num', 'counts_red_up', 'counts_red_down', 'counts_red_left', 'counts_red_right', 'counts_blue_up', 'counts_blue_down', 'counts_blue_left', 'counts_blue_right']
    for col in int_cols:
        df[col] = df[col].astype('Int64')
        

    ## add info on button press counts
    df['counts_red'] = df['counts_red_up'] + df['counts_red_down'] + df['counts_red_left'] + df['counts_red_right']
    df['counts_blue'] = df['counts_blue_up'] + df['counts_blue_down'] + df['counts_blue_left'] + df['counts_blue_right']
    df['counts_diff'] = np.abs(df['counts_red'] - df['counts_blue'])
    df['counts_post_red'] = df['counts_post_red_up'] + df['counts_post_red_down'] + df['counts_post_red_left'] + df['counts_post_red_right']
    df['counts_post_blue'] = df['counts_post_blue_up'] + df['counts_post_blue_down'] + df['counts_post_blue_left'] + df['counts_post_blue_right']
    df['counts_post_diff'] = np.abs(df['counts_post_red'] - df['counts_post_blue'])

    df['chosen_counts'] = np.nan
    df['unchosen_counts'] = np.nan
    df.loc[df['chosen_button'] == 'red', 'chosen_counts'] = df['counts_red']
    df.loc[df['chosen_button'] == 'blue', 'chosen_counts'] = df['counts_blue']
    df.loc[df['chosen_button'] == 'red', 'unchosen_counts'] = df['counts_blue']
    df.loc[df['chosen_button'] == 'blue', 'unchosen_counts'] = df['counts_red']
    df['chosen_counts_diff'] = df['chosen_counts'] - df['unchosen_counts']
    
    df['least_sampled'] = np.nan
    df.loc[df['counts_red'] < df['counts_blue'], 'least_sampled'] = 'red'
    df.loc[df['counts_blue'] < df['counts_red'], 'least_sampled'] = 'blue'
    df.loc[df['counts_red'] == df['counts_blue'], 'least_sampled'] = 'equal'
    df['most_sampled'] = np.nan
    df.loc[df['counts_red'] > df['counts_blue'], 'most_sampled'] = 'red'
    df.loc[df['counts_blue'] > df['counts_red'], 'most_sampled'] = 'blue'
    df.loc[df['counts_red'] == df['counts_blue'], 'most_sampled'] = 'equal'

    df['chose_least_sampled'] = np.nan
    df.loc[df['chosen_button'] == df['least_sampled'], 'chose_least_sampled'] = True
    df.loc[(df['chosen_button'] != df['least_sampled']) & (df['least_sampled']!='equal'), 'chose_least_sampled'] = False
    df['chose_most_sampled'] = np.nan
    df.loc[df['chosen_button'] == df['most_sampled'], 'chose_most_sampled'] = True
    df.loc[(df['chosen_button'] != df['most_sampled']) & (df['most_sampled']!='equal'), 'chose_most_sampled'] = False

    ## choice repeats
    df['repeat_choice'] = np.nan
    df['repeat_choice'] = df['chosen_button'] == df['chosen_button'].shift(1)
    df.loc[df['trial_num'] == 1, 'repeat_choice'] = np.nan

    ## entropy under dirichlet
    alpha = df['alpha'].values[0]
    df['entropy_red'] = df.apply(lambda x: dirichlet_entropy(np.array([x['counts_red_up']+alpha, x['counts_red_left'] +alpha, x['counts_red_down'] +alpha, x['counts_red_right']+alpha])), axis=1)
    df['entropy_blue'] = df.apply(lambda x: dirichlet_entropy(np.array([x['counts_blue_up']+alpha, x['counts_blue_left'] +alpha, x['counts_blue_down'] +alpha, x['counts_blue_right']+alpha])), axis=1)
    df['total_entropy'] = df['entropy_red'] + df['entropy_blue']
    df['entropy_chosen'] = np.nan
    df.loc[df['chosen_button'] == 'red', 'entropy_chosen'] = df['entropy_red']
    df.loc[df['chosen_button'] == 'blue', 'entropy_chosen'] = df['entropy_blue']
    df['entropy_unchosen'] = np.nan
    df.loc[df['chosen_button'] == 'red', 'entropy_unchosen'] = df['entropy_blue']
    df.loc[df['chosen_button'] == 'blue', 'entropy_unchosen'] = df['entropy_red']


    ## info on number of different outcomes observed for each button
    df['n_diff_outcomes_red'] = df[['counts_red_up', 'counts_red_down', 'counts_red_left', 'counts_red_right']].gt(0).sum(axis=1)
    df['n_diff_outcomes_blue'] = df[['counts_blue_up', 'counts_blue_down', 'counts_blue_left', 'counts_blue_right']].gt(0).sum(axis=1)
    df['n_diff_outcomes_chosen'] = np.nan
    df['n_diff_outcomes_unchosen'] = np.nan
    df.loc[df['chosen_button'] == 'red', 'n_diff_outcomes_chosen'] = df['n_diff_outcomes_red']
    df.loc[df['chosen_button'] == 'blue', 'n_diff_outcomes_chosen'] = df['n_diff_outcomes_blue']
    df.loc[df['chosen_button'] == 'red', 'n_diff_outcomes_unchosen'] = df['n_diff_outcomes_blue']
    df.loc[df['chosen_button'] == 'blue', 'n_diff_outcomes_unchosen'] = df['n_diff_outcomes_red']


    ### get canonical history

    ## convert counts to n_arms x n_outcomes arrays
    n_arms = 2
    n_outcomes = 4
    df['counts_array'] = df.apply(lambda x: np.array([
        [x['counts_blue_up'], x['counts_blue_down'], x['counts_blue_left'], x['counts_blue_right']],
        [x['counts_red_up'], x['counts_red_down'], x['counts_red_left'], x['counts_red_right']],
                                                        ]), axis=1)
    df['canonical_counts_array'] = df['counts_array'].apply(lambda x: canonical_count_matrix(x)[0])
    df['history_str'] = df['canonical_counts_array'].apply(lambda x: array_to_hist(x, n_arms, n_outcomes)[1])

    ## map history str back onto colours and outcomes
    df = df.apply(lambda x: canon_to_concrete(x), axis=1)

    ## map strings onto numbers
    button_map = {'blue': 0, 'red': 1}
    outcome_map = {'up': 0, 'left': 1, 'down': 2, 'right': 3}
    df['chosen_button'] = df['chosen_button'].map(button_map)
    df['outcome'] = df['outcome'].map(outcome_map)

    ##0-based indexing for room and trial
    df['room_num'] = df['room_num'] - 1
    df['trial_num'] = df['trial_num'] - 1

    ## some renaming of cols
    df.rename(columns={'trial_num': 'trial',
                       'room_num': 'room',
                       'chosen_button': 'action',
                       'ended_early': 'terminated',
        }, inplace=True)

    return df


def dirichlet_entropy(alphas):
    a0 = np.sum(alphas)
    log_beta_a = np.log(np.prod(gamma(alphas)) / gamma(a0))
    entropy = log_beta_a + (a0 - len(alphas)) * digamma(a0) - np.sum((alphas - 1) * digamma(alphas))
    return entropy


def _coin_df(trials, session):
    """One row per room: the gold-coin choice made after sampling.

    `collected_gold` is the running total across rooms, not a per-room outcome;
    `success` is whether this room's coin was won. `chosen_gold_prob` and
    `best_gold_prob` are true probabilities of the required outcome, so `optimal`
    and `regret` score the choice against ground truth, not against the
    participant's displayed belief (`post_*`).
    """
    rows = []
    for t in _by_task(trials, "gold"):
        row = dict(session)
        row.update(
            room_num=t.get("room_num"),
            chosen_button=t.get("chosen_button"),
            gold_outcome=t.get("gold_outcome"),
            rt=t.get("rt"),
            success=t.get("success"),
            collected_gold=t.get("collected_gold"),
            outcome_shown=t.get("outcome_shown"),
            chosen_gold_prob=t.get("chosen_gold_prob"),
            best_gold_prob=t.get("best_gold_prob"),
            chosen_button_ctx=t.get("chosen_button_ctx"),
            trial_index=t.get("trial_index"),
            time_elapsed=t.get("time_elapsed"),
        )
        row.update(_flatten(t.get("counts"), "counts"))         # belief the choice used
        row.update(_flatten(t.get("posterior_means"), "post"))
        for button, ctx in (t.get("button_ctx") or {}).items():
            row[f"ctx_{button}"] = ctx
        row["chosen_button_gen_alpha"] = _gen_alpha(session, t.get("chosen_button_ctx"))
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["chosen_button_pos"] = np.where(df.chosen_button == df.button_upper, "upper", "lower")
        df["optimal"] = np.isclose(df.chosen_gold_prob, df.best_gold_prob)
        df["regret"] = df.best_gold_prob - df.chosen_gold_prob

    return df


def _attention_df(trials, session):
    """One row per attention check.

    n_seen/n_correct/accuracy are running values; the last row gives the totals.
    """
    rows = []
    for t in _by_task(trials, "attention_check"):
        row = dict(session)
        row.update(
            trial_index=t.get("trial_index"),
            rt=t.get("rt"),
            probed_button=t.get("attention_color"),
            correct_outcome=t.get("attention_correct_outcome"),
            chosen_outcome=t.get("attention_chosen_outcome"),
            is_correct=t.get("attention_is_correct"),
            n_seen=t.get("attention_total"),
            n_correct=t.get("attention_correct_count"),
            accuracy=t.get("attention_accuracy"),
        )
        row.update(_flatten(t.get("attention_counts"), "counts"))
        rows.append(row)
    return pd.DataFrame(rows)


def _skip_index(scale):
    """Index of the "Skip the question" option in a scale, or None.

    Stored responses are the scale indices themselves: the plugin builds its
    values from `scoring_index` (default 0, and no questionnaire overrides it),
    so the option's position *is* the value recorded.
    """
    for i, label in enumerate(scale):
        if "skip" in str(_strip_html(label)).lower():
            return i
    return None


def _questionnaire_df(trials, session, task):
    """One row per item for a Likert questionnaire (IUS, LOC1, LOC2).

    `response` is 0-indexed (0 = leftmost scale option) and NaN when the item was
    skipped; `response_label` gives the chosen option verbatim. Note IUS-12 and
    the IPC LOC scales are conventionally scored 1-indexed, so add 1 to score.

    No questionnaire sets the plugin's `reverse` parameter, so nothing here is
    reverse-keyed -- these are raw scale positions.
    """
    ts = _by_task(trials, task)
    if not ts:
        return pd.DataFrame()
    t = ts[0]

    items = t.get("items") or []
    scale = t.get("scale") or []
    item_order = list(t.get("item_order") or range(len(items)))
    infrequency = t.get("infrequency_items") or [False] * len(items)
    ipc = t.get("IPC")
    skip = _skip_index(scale)

    click_ids = t.get("radio_event_ids") or []
    click_times = t.get("radio_event_times") or []

    rows = []
    for qid, value in sorted((t.get("responses") or {}).items()):
        # The plugin names each radio Q{item_order[i]+1}, i.e. from the item's
        # ORIGINAL index, not its display position -- so Q01 is always items[0]
        # however the items were shuffled.
        idx = int(re.sub(r"\D", "", qid)) - 1
        value = int(value)
        skipped = skip is not None and value == skip
        clicks = [ms for q, ms in zip(click_ids, click_times) if q == qid]

        row = dict(session)
        row.update(
            task=task,
            qid=qid,
            item_index=idx,
            item=_strip_html(items[idx]) if idx < len(items) else None,
            display_position=item_order.index(idx) if idx in item_order else None,
            response=np.nan if skipped else value,
            response_raw=value,
            response_label=_strip_html(scale[value]) if value < len(scale) else None,
            skipped=skipped,
            is_infrequency=bool(infrequency[idx]) if idx < len(infrequency) else False,
            n_clicks=len(clicks),          # >1 means the answer was revised
            first_click_ms=clicks[0] if clicks else np.nan,
            last_click_ms=clicks[-1] if clicks else np.nan,
            survey_rt=t.get("rt"),
            straightlining=t.get("straightlining"),
            zigzagging=t.get("zigzagging"),
            honeypot=t.get("honeypot"),
        )
        if ipc:
            row["IPC"] = ipc[idx] if idx < len(ipc) else None
        rows.append(row)

    return pd.DataFrame(rows).sort_values("item_index", ignore_index=True)


def _feedback_df(trials, session):
    """One row per participant: the free-text/debrief responses.

    Columns follow whichever fields the plugin was configured to show, so
    enabling e.g. the workload items later widens this automatically.
    """
    rows = []
    for t in _by_task(trials, "expfeedback"):
        row = dict(session)
        row.update(t.get("responses") or {})
        row["rt"] = t.get("rt")
        rows.append(row)
    return pd.DataFrame(rows)


#----------------------------------------------------------------------------#
# Public API
#----------------------------------------------------------------------------#

def load_participant(path):
    """Load one participant's JSON file into a dataframe per task phase.

    Args:
        path: path to a participant's .json file.

    Returns:
        dict of dataframes keyed by phase: "meta", "rooms", "sample", "coin",
        "attention", "IUS", "LOC1", "LOC2", "feedback". Phases the participant
        did not reach come back as empty dataframes.
    """
    trials = _read_trials(path)
    session = _session(trials, path)

    dfs = {
        "meta": _meta_df(trials, session),
        "rooms": _rooms_df(trials, session),
        "sample": _sample_df(trials, session),
        "coin": _coin_df(trials, session),
        "attention": _attention_df(trials, session),
    }
    for q in QUESTIONNAIRES:
        dfs[q] = _questionnaire_df(trials, session, q)
    dfs["feedback"] = _feedback_df(trials, session)
    return dfs


def load_directory(data_dir, pattern="*.json"):
    """Load every participant file in a directory, concatenated per phase.

    Args:
        data_dir: directory of participant .json files.
        pattern: glob for the files to include.

    Returns:
        dict of dataframes keyed by phase, as load_participant, with all
        participants stacked. Use `source_file` or `subject_id` to split them.
    """
    paths = sorted(Path(data_dir).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files matching {pattern!r} in {data_dir}")
    
    ## remove pid_map.json from paths
    paths = [p for p in paths if p.name != 'pid_map.json']

    loaded = [load_participant(p) for p in paths]
    out = {}
    for phase in PHASES:
        parts = [d[phase] for d in loaded if not d[phase].empty]
        out[phase] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return out


## get bonus info
def load_reversed_pid_map(df):
    json_path = 'expt/Experiment4/data/pilot/pid_map.json'
    with open(json_path, 'r') as f:
        pid_map = json.load(f)  # prolific_id -> subject_id
    
    # reverse: subject_id -> prolific_id
    reversed_map = {subject_id: prolific_id for prolific_id, subject_id in pid_map.items() if subject_id in df['subject_id'].values}

    return reversed_map

## print bonuses
def print_bonuses(df):
    for subject_id in df['subject_id'].unique():
        bonus = df.loc[df['subject_id'] == subject_id, 'bonus_gbp'].values[0]
        if bonus is not None:
            print(subject_id+','+str(bonus))