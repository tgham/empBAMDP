//----------------------------------------------------------------------------//
// DEBUGGING: jump straight to the rooms -- skips the instructions, the
// comprehension check, and the attention-check intro/practice. The in-task
// attention checks still run, as does the consent form and the fullscreen
// prompt. The comprehension gate is treated as passed (everything downstream is
// conditional on it), and the Prolific redirect at the end is suppressed.
// MUST be false for any real run: it bypasses the comprehension gate.
//----------------------------------------------------------------------------//
const DEBUGGING = true;

//----------------------------------------------------------------------------//
// Experiment parameters
//----------------------------------------------------------------------------//
// N=3 buttons, K=4 outcomes (cardinal directions), T=8 trials, Dirichlet alpha
const N_BUTTONS = 3;
const K_OUTCOMES = 4;
const N_TRIALS = 8;   // total presses a room is worth: the observed (preset) presses
                      // plus the participant's own. Also sets the token capacity of a
                      // counter cell and the denominator of the default press cost.

// How many free choices the participant gets AFTER watching the preset history.
const N_REMAINING_TRIALS = 1;

// Cost of a single press, as a fraction of the room's gold coin.
// Set to 0 for a cost-free version: nothing depletes, in the demo or the choice.
const SAMPLE_COST = 1 / (N_TRIALS + 1);

// Do the presses the participant WATCHES also spend the coin? true is the faithful
// version (the model charges for every trial in the history); false gives the
// participant a full coin at the moment of their own choice.
const DEPLETE_DURING_DEMO = false;

// Pacing of the observation phase, in ms. press pulse -> agent moves -> token lands
// -> pause on the outcome -> return to centre -> pause before the next press.
const DEMO_PRESS_MS = 450;    // button pulse before the agent moves
const DEMO_VIEW_MS = 800;     // time spent looking at the reached location
const DEMO_GAP_MS = 550;      // pause back at the centre before the next press

// N_ROOMS is not a constant any more: it is the number of preset histories in
// presets.json, and so is only known once that file has been fetched. See
// loadPresets() below -- everything downstream of it must run inside that promise.
let N_ROOMS = 0;

// CONTEXTUAL controls which prior each button's hidden transition function is
// drawn from at the start of a room:
//   false -> every button is drawn from Dirichlet(ALPHA), as before.
//   true  -> each button independently gets a fair coin flip between
//            Dirichlet(ALPHA_CTX1) and Dirichlet(ALPHA_CTX2), so the two buttons
//            in a room can come from different priors. The context drawn is
//            recorded in BUTTON_CTX and logged, but is never cued to the
//            participant.
const ALPHA = 0.4;
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
const BUTTONS = ["red", "blue", "green"];
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
// one triangle layout per room; filled in by loadPresets() once N_ROOMS is known
let ROOM_BUTTON_ORDERS = [];

//----------------------------------------------------------------------------//
// Preset observation histories
//----------------------------------------------------------------------------//
// presets.json is a list of n_rooms CANONICAL histories, as written by the
// modelling notebook. Each history is a list of [[action, outcome], count] pairs
// over canonical (smallest-first) labels, e.g.
//     [[[0, 0], 5], [[1, 0], 2]]
// = canonical action 0 led to canonical outcome 0 five times, action 1 to
// outcome 0 twice. Actions/outcomes never observed simply do not appear.
//
// Because the labels are canonical, every room draws its own random
// action -> colour and outcome -> direction mapping, so the same diagnostic
// history looks different each time it recurs.
let PRESETS = null;      // the raw canonical histories
let ROOM_PRESETS = [];   // one concretised entry per room (see buildRoomPreset)

// Expand one canonical history into everything a room needs.
function buildRoomPreset(history) {
    // canonical action i -> BUTTONS[perm[i]], canonical outcome k -> OUTCOMES[perm[k]]
    const buttonPerm = shuffled(BUTTONS);
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

// Fetch presets.json and derive N_ROOMS, the per-room button layouts and the
// per-room concretised presets. index.html builds its timeline inside this promise.
function loadPresets() {
    return fetch("presets.json")
        .then(function (resp) {
            if (!resp.ok) throw new Error("presets.json: HTTP " + resp.status);
            return resp.json();
        })
        .then(function (data) {
            if (!Array.isArray(data) || data.length === 0) {
                throw new Error("presets.json must be a non-empty list of histories");
            }
            PRESETS = data;
            N_ROOMS = PRESETS.length;
            ROOM_BUTTON_ORDERS = Array.from({ length: N_ROOMS }, randomButtonOrder);
            ROOM_PRESETS = PRESETS.map(buildRoomPreset);
            // a preset must leave room for the participant's own choices
            const longest = Math.max.apply(null, ROOM_PRESETS.map(p => p.n_preset_presses));
            if (longest + N_REMAINING_TRIALS > N_TRIALS) {
                console.warn(
                    `longest preset history (${longest}) + N_REMAINING_TRIALS ` +
                    `(${N_REMAINING_TRIALS}) exceeds N_TRIALS (${N_TRIALS}): the coin ` +
                    `will run out and counter cells may overflow.`
                );
            }
            return ROOM_PRESETS;
        });
}

//----------------------------------------------------------------------------//
// Belief display mode:
//   "overlay"  -> both buttons' posteriors shown in the main grid, each reachable
//                 cell split diagonally (red upper-right triangle, blue lower-left).
//   "separate" -> a dedicated belief grid per button, to the right of the room.
//   "counters" -> a single grid; each reachable cell has a red half and a blue
//                 half, each filling with up to N_TRIALS tokens as that button is
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
