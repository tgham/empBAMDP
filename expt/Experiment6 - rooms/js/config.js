//----------------------------------------------------------------------------//
// DEBUGGING: jump straight to the rooms -- skips the instructions, the
// comprehension check, and the attention-check intro/practice. The in-task
// attention checks still run, as does the consent form and the fullscreen
// prompt. The comprehension gate is treated as passed (everything downstream is
// conditional on it), and the Prolific redirect at the end is suppressed.
// MUST be false for any real run: it bypasses the comprehension gate.
//----------------------------------------------------------------------------//
const DEBUGGING = false;

//----------------------------------------------------------------------------//
// Experiment parameters
//----------------------------------------------------------------------------//
// N buttons per room (2 or 3), K=4 outcomes (cardinal directions), Dirichlet alpha
const N_BUTTONS = 2;
const K_OUTCOMES = 4;
// N_AFC: how many rooms are offered on each trial. The participant watches every
// room's preset history, then clicks the one they want to enter for the gold.
const N_AFC = 2;
const N_TRIALS = 8;   // total presses a room is worth: the observed (preset) presses
                      // plus the participant's own. Also the denominator of the
                      // default press cost.

// Token slots per button in a counter cell, which sets the token spacing. Set by
// loadPresets() to the most times any one button-location pairing is observed in
// the presets file (plus the participant's own presses), so the fullest cell is
// exactly full. The instructions use the same spacing throughout.
let COUNTER_SLOTS = N_TRIALS;

// How many free choices the participant gets AFTER watching the preset history.
// None in this design: they choose a room instead of pressing in it.
const N_REMAINING_TRIALS = 0;

// Cost of a single press, as a fraction of the room's gold coin.
// Set to 0 for a cost-free version: nothing depletes, in the demo or the choice.
// const SAMPLE_COST = 1 / (N_TRIALS + 1);
const SAMPLE_COST = 0;
// const SAMPLE_COST = 1/64;

// TERMINATE: does this version of the design include the termination arm -- the
// orange tick that ends testing early and moves straight on to the gold?
//   true  -> the tick is drawn beside the room, explained in the instructions and
//            quizzed in the comprehension check, exactly as before.
//   false -> there is no tick anywhere: not in the task, not on any instruction
//            slide, and the tick question is dropped from the quiz. The room is
//            drawn without it (rather than behind an invisible placeholder), so
//            it sits centred -- consistent across every screen, since TERMINATE
//            is fixed for the whole session.
// It must match the model the presets were generated under -- see PRESETS_FILE,
// which picks the Termination / noTermination file off this flag.
const TERMINATE = false;

// Do the presses the participant WATCHES also spend the coin? true is the faithful
// version (the model charges for every trial in the history); false gives the
// participant a full coin at the moment of their own choice.
const DEPLETE_DURING_DEMO = false;

// Pacing of the observation phase, in ms. press pulse -> agent moves -> token lands
// -> pause on the outcome -> return to centre -> pause before the next press.
const DEMO_PRESS_MS = 200;    // button pulse before the agent moves
const DEMO_VIEW_MS = 300;     // time spent looking at the reached location
const DEMO_GAP_MS = 200;      // pause back at the centre before the next press
// The choice screen: how long the empty rooms are shown before the first room's
// presses start, and the pause between one room's presses and the next room's.
const CHOICE_PREVIEW_MS = 500;
const CHOICE_ROOM_GAP_MS = 800;

// N_ROOMS is not a constant any more: it is the number of preset histories in the
// presets file, and so is only known once that file has been fetched. See
// PRESETS_FILE / loadPresets() below -- everything downstream of loadPresets must
// run inside that promise.
let N_ROOMS = 0;

// CONTEXTUAL controls which prior each button's hidden transition function is
// drawn from at the start of a room:
//   false -> every button is drawn from Dirichlet(ALPHA), as before.
//   true  -> each button independently gets a fair coin flip between
//            Dirichlet(ALPHA_CTX1) and Dirichlet(ALPHA_CTX2), so the two buttons
//            in a room can come from different priors. The context drawn is
//            recorded in BUTTON_CTX and logged, but is never cued to the
//            participant.
const ALPHA = 0.25;
const CONTEXTUAL = false;
const ALPHA_CTX1 = 0.25; // context 1 prior
const ALPHA_CTX2 = 1; // context 2 prior

// After the sampling trials, a gold coin appears at a random reachable cell and
// the participant picks a button to try to reach it.
//   true  -> reveal the outcome: the agent moves per the transition function and
//            we show whether the gold was obtained.
//   false -> do not reveal the outcome; move straight on to the next room.
// In the real experiment we do NOT reveal it (participants never learn whether
// they reached the coin). The instruction gold DEMOS reveal it regardless, for
// teaching, and explain that the real rooms keep it hidden.
const SHOW_GOLD_OUTCOME = false;

// OBJECTIVE: what the room choice is for.
//   "gold"    -> "Click on the room you want to enter"; the gold selection task
//                then runs in the chosen room.
//   "control" -> "In which room do you feel the greatest sense of control?"; no
//                gold selection task -- after the choice (and the confidence
//                rating, if CONFIDENCE) it is straight on to the next trial.
// const OBJECTIVE = "gold";
const OBJECTIVE = "control";
if (OBJECTIVE !== "gold" && OBJECTIVE !== "control") {
    throw new Error(`OBJECTIVE must be "gold" or "control", not ${JSON.stringify(OBJECTIVE)}`);
}

// CONFIDENCE: after choosing a room, is the participant asked how likely they
// think they are to get a coin in it?
//   true  -> the chosen room is shown on its own with a 0-100 slider (see
//            make_confidence_trial); logged as task "confidence". The question
//            is worded for the OBJECTIVE.
//   false -> straight on from the choice to the gold (or, under "control", to
//            the next trial).
const CONFIDENCE = false;

// SHUFFLE_TRIALS: is the order of the trials in the presets file shuffled per
// participant?
//   false -> trials run in file order (the file is already shuffled when the
//            notebook writes it).
//   true  -> a fresh random order per participant.
// Either way each trial logs `preset_trial_index`, its position in the file.
const SHUFFLE_TRIALS = false;

// ATTENTION_CHECK_EVERY: an attention-check block is run after every this many
// trials (never after the final one). 0 -> no attention checks in the task.
const ATTENTION_CHECK_EVERY = 2;

// Prolific redirect links (fill in later). If left blank, no redirect happens.
const REDIRECT_COMPLETE = "https://app.prolific.com/submissions/complete?cc=COKTR2G5"; // shown after finishing the experiment
const REDIRECT_FAIL = "https://app.prolific.com/submissions/complete?cc=C908OCW9";     // shown after failing the comprehension check twice

//----------------------------------------------------------------------------//
// Grid geometry
//----------------------------------------------------------------------------//
// 3x3 cell indices:   0 1 2
//                     3 4 5
//                     6 7 8
// centre = 4; the four cardinal outcomes live on the edge-centre cells,
// corners (0,2,6,8) are never reached.
const OUTCOMES = ["up", "right", "down", "left"];
const OUTCOME_CELL = { up: 1, right: 5, down: 7, left: 3 };

//----------------------------------------------------------------------------//
// Buttons
//----------------------------------------------------------------------------//
// 2 buttons -> red/blue (drawn on a diagonal); 3 buttons -> add green (triangle)
const BUTTONS = ["red", "blue", "green"].slice(0, N_BUTTONS);
const BTN_COLOR = { red: "220,40,40", blue: "40,90,220", green: "40,170,70" };

// Fisher-Yates, on a copy.
function shuffled(arr) {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
}
function randomButtonOrder() {
    return shuffled(BUTTONS);
}
let BUTTON_ORDER = randomButtonOrder();
// the button layout of each room offered on each trial: ROOM_BUTTON_ORDERS[t][option]
// (the same for every room of a trial); filled in by loadPresets() once N_ROOMS is known
let ROOM_BUTTON_ORDERS = [];
// left-to-right screen placement of each trial's options: OPTION_ORDERS[t][position]
// = option index in the presets file, so a preset's index is not its screen side
let OPTION_ORDERS = [];
// TRIAL_ORDER[t] = index in the presets file of the t-th trial run (the identity
// unless SHUFFLE_TRIALS)
let TRIAL_ORDER = [];
// the room the participant picked on the current trial, set by make_room_choice
// and read by the gold trial: { option, position, preset, buttonOrder }
let CHOSEN = null;

//----------------------------------------------------------------------------//
// Preset observation histories
//----------------------------------------------------------------------------//
// The presets file is a list of n_rooms CANONICAL histories, as written by the
// modelling notebook, named for the arity it was generated at (see PRESETS_FILE
// below). Each history is a list of [[action, outcome], count] pairs
// over canonical (smallest-first) labels, e.g.
//     [[[0, 0], 5], [[1, 0], 2]]
// = canonical action 0 led to canonical outcome 0 five times, action 1 to
// outcome 0 twice. Actions/outcomes never observed simply do not appear.
//
// Because the labels are canonical, every trial draws its own random
// action -> colour mapping (shared by its rooms) and every room its own
// outcome -> direction mapping, so the same diagnostic history looks different
// each time it recurs.
//
// In this design the file is one level deeper: a list of TRIALS, each a list of
// N_AFC histories (one per room offered). A history may be empty -- a room with
// no observations at all.
let PRESETS = null;      // the raw canonical histories, PRESETS[t][option]
let ROOM_PRESETS = [];   // ROOM_PRESETS[t][option], concretised (see buildRoomPreset)

// Expand one canonical history into everything a room needs. `buttonPerm` fixes
// the action -> colour mapping, so all rooms of a trial can share it; left out,
// the room draws its own.
function buildRoomPreset(history, buttonPerm) {
    // canonical action i -> BUTTONS[perm[i]], canonical outcome k -> OUTCOMES[perm[k]]
    buttonPerm = buttonPerm || shuffled(BUTTONS);
    const outcomePerm = shuffled(OUTCOMES);
    const buttonMap = {};
    buttonPerm.forEach((b, i) => { buttonMap[i] = b; });
    const outcomeMap = {};
    outcomePerm.forEach((o, k) => { outcomeMap[k] = o; });

    // concrete tally, and the flat list of presses that produced it
    const presetCounts = freshButtonMap();
    const sequence = [];
    for (const entry of history) {
        const a = entry[0][0];
        const o = entry[0][1];
        const n = entry[1];
        const button = buttonMap[a];
        const outcome = outcomeMap[o];
        if (button === undefined || outcome === undefined) {
            console.error("preset references an out-of-range action/outcome", entry);
            continue;
        }
        presetCounts[button][outcome] += n;
        for (let i = 0; i < n; i++) sequence.push({ button: button, outcome: outcome });
    }

    return {
        history: history,                   // logged verbatim, for analysis
        button_map: buttonMap,
        outcome_map: outcomeMap,
        presetCounts: presetCounts,
        // the history only fixes COUNTS, not an order: shuffle so the observed
        // presses interleave rather than arriving grouped by button
        sequence: shuffled(sequence),
        n_preset_presses: sequence.length
    };
}

// The presets file carries the arity it was generated for in its NAME:
// `{n_arms}arms_{n_outcomes}outcomes_{n_AFC}AFC_presets.json`. We build the same
// name from the configured design, so N_BUTTONS/K_OUTCOMES/N_AFC pick the matching
// file and a mismatch shows up as a 404 rather than as histories silently mapped
// onto the wrong number of buttons, locations or rooms.
const PRESETS_FILE = `${N_BUTTONS}arms_${K_OUTCOMES}outcomes_${N_AFC}AFC_presets.json`;

// Reject anything the display could not honestly show: a history referring to a
// button or a location this design does not have. `BUTTONS` and `OUTCOMES` are what
// actually get drawn, so they are the real bound, not N_BUTTONS/K_OUTCOMES.
function validatePresets(data) {
    if (!Array.isArray(data) || data.length === 0) {
        throw new Error(`${PRESETS_FILE} must be a non-empty list of trials`);
    }
    if (BUTTONS.length !== N_BUTTONS || OUTCOMES.length !== K_OUTCOMES) {
        throw new Error(
            `design mismatch: N_BUTTONS=${N_BUTTONS}/K_OUTCOMES=${K_OUTCOMES} but ` +
            `${BUTTONS.length} button colours and ${OUTCOMES.length} locations are ` +
            `defined. The grid geometry and the button triangle are drawn from ` +
            `BUTTONS/OUTCOMES, so both must be updated together.`
        );
    }
    let maxAction = -1, maxOutcome = -1;
    data.forEach(function (trial, t) {
        if (!Array.isArray(trial) || trial.length !== N_AFC) {
            throw new Error(`${PRESETS_FILE}: trial ${t} is not a list of ${N_AFC} histories`);
        }
        trial.forEach(function (history, j) {
            // an empty history is allowed: a room with nothing observed yet
            if (!Array.isArray(history)) {
                throw new Error(`${PRESETS_FILE}: trial ${t} room ${j} is not a history`);
            }
            for (const entry of history) {
                const a = entry[0][0], o = entry[0][1], n = entry[1];
                if (!Number.isInteger(a) || !Number.isInteger(o) || !Number.isInteger(n) || n < 1) {
                    throw new Error(`${PRESETS_FILE}: trial ${t} room ${j} has a malformed entry ${JSON.stringify(entry)}`);
                }
                if (a >= N_BUTTONS || o >= K_OUTCOMES || a < 0 || o < 0) {
                    throw new Error(
                        `${PRESETS_FILE}: trial ${t} room ${j} uses action ${a} / outcome ${o}, outside ` +
                        `the ${N_BUTTONS}-button, ${K_OUTCOMES}-location design this file is named for`
                    );
                }
                if (a > maxAction) maxAction = a;
                if (o > maxOutcome) maxOutcome = o;
            }
        });
    });
    // not an error -- canonical histories legitimately leave later arms/outcomes
    // unobserved -- but worth seeing in the console when checking a new preset set
    console.log(
        `${PRESETS_FILE}: ${data.length} trials of ${N_AFC} rooms, actions 0-${maxAction} of ${N_BUTTONS}, ` +
        `outcomes 0-${maxOutcome} of ${K_OUTCOMES}`
    );
}

// N_TRIALS is the room's press budget: with SAMPLE_COST = 1/(N_TRIALS+1), a room
// worth more than N_TRIALS presses spends the whole coin before the participant
// chooses. A warning rather than an error: a preset set can legitimately be paired
// with a larger N_TRIALS, and this tells you when it has to be.
//
// Also sizes the counters: COUNTER_SLOTS = the fullest (button, location) cell in
// any room, plus the participant's own presses, which can all land on it.
function checkPresetCapacity() {
    const allRooms = ROOM_PRESETS.flat();
    const longest = Math.max.apply(null, allRooms.map(p => p.n_preset_presses));
    let fullestCell = 0;
    allRooms.forEach(function (p) {
        for (const b of BUTTONS) {
            for (const o of OUTCOMES) {
                if (p.presetCounts[b][o] > fullestCell) fullestCell = p.presetCounts[b][o];
            }
        }
    });
    COUNTER_SLOTS = Math.max(1, fullestCell + N_REMAINING_TRIALS);
    if (longest + N_REMAINING_TRIALS > N_TRIALS) {
        console.warn(
            `${PRESETS_FILE}: longest history (${longest}) + N_REMAINING_TRIALS ` +
            `(${N_REMAINING_TRIALS}) = ${longest + N_REMAINING_TRIALS} exceeds ` +
            `N_TRIALS (${N_TRIALS}). With a non-zero SAMPLE_COST the coin runs out ` +
            `before the participant chooses; raise N_TRIALS to at least ` +
            `${longest + N_REMAINING_TRIALS}.`
        );
    }
}

// Fetch the presets file and derive N_ROOMS (here: the number of trials, each a
// choice between N_AFC rooms), and per trial the rooms' button layouts, their
// concretised presets and their screen order. index.html builds its timeline
// inside this promise.
function loadPresets() {
    return fetch(PRESETS_FILE)
        .then(function (resp) {
            if (!resp.ok) throw new Error(`${PRESETS_FILE}: HTTP ${resp.status}`);
            return resp.json();
        })
        .then(function (data) {
            validatePresets(data);
            TRIAL_ORDER = data.map((_, i) => i);
            if (SHUFFLE_TRIALS) TRIAL_ORDER = shuffled(TRIAL_ORDER);
            // PRESETS (and everything derived from it below) is in run order
            PRESETS = TRIAL_ORDER.map(i => data[i]);
            N_ROOMS = PRESETS.length;
            // the rooms of a trial share one button layout and one action -> colour
            // mapping (e.g. a0 is blue and upper-left in every room), so they differ
            // only in their histories and outcome -> direction mappings; both are
            // redrawn per trial
            ROOM_BUTTON_ORDERS = PRESETS.map(function (trial) {
                const order = randomButtonOrder();
                return trial.map(() => order);
            });
            ROOM_PRESETS = PRESETS.map(function (trial) {
                const buttonPerm = shuffled(BUTTONS);
                return trial.map(history => buildRoomPreset(history, buttonPerm));
            });
            OPTION_ORDERS = PRESETS.map(trial => shuffled(trial.map((h, j) => j)));
            checkPresetCapacity();
            return ROOM_PRESETS;
        });
}

//----------------------------------------------------------------------------//
// Belief display mode:
//   "overlay"  -> both buttons' posteriors shown in the main grid, each reachable
//                 cell split diagonally (red upper-right triangle, blue lower-left).
//   "separate" -> a dedicated belief grid per button, to the right of the room.
//   "counters" -> a single grid; each reachable cell has a red half and a blue
//                 half, each filling with up to COUNTER_SLOTS tokens as that button is
//                 observed leading there (a running tally, not a probability).
//----------------------------------------------------------------------------//
// const BELIEF_DISPLAY = "overlay";
// const BELIEF_DISPLAY = "separate";
const BELIEF_DISPLAY = "counters";

function freshOutcomeMap(fillFn) {
    const m = {};
    for (const o of OUTCOMES) m[o] = fillFn ? fillFn() : 0;
    return m;
}
function freshButtonMap(fillFn) {
    const m = {};
    for (const b of BUTTONS) m[b] = fillFn ? fillFn() : freshOutcomeMap();
    return m;
}

let TRUE_T = freshButtonMap(() => freshOutcomeMap(() => 1 / OUTCOMES.length));
let BUTTON_CTX = Object.fromEntries(BUTTONS.map(b => [b, null]));
let counts = freshButtonMap();

// running total of gold coins collected across rooms
let collected_gold = 0;

// set true when the participant ends sampling early (via the tick button); reset
// at the start of each room so the remaining sampling trials are skipped.
let sampling_ended = false;
