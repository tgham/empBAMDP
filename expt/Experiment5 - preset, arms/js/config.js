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
// N=3 buttons, K=4 outcomes (cardinal directions), T=8 trials, Dirichlet alpha
const N_BUTTONS = 3;
const K_OUTCOMES = 4;
const N_TRIALS = 10;   // total presses a room is worth: the observed (preset) presses
                      // plus the participant's own. Also sets the token capacity of a
                      // counter cell and the denominator of the default press cost.

// How many free choices the participant gets AFTER watching the preset history.
const N_REMAINING_TRIALS = 1;

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
const DEMO_PRESS_MS = 300;    // button pulse before the agent moves
const DEMO_VIEW_MS = 500;     // time spent looking at the reached location
const DEMO_GAP_MS = 300;      // pause back at the centre before the next press

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
// The presets file is a list of n_rooms CANONICAL histories, as written by the
// modelling notebook, named for the arity it was generated at (see PRESETS_FILE
// below). Each history is a list of [[action, outcome], count] pairs
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

// The presets file carries the arity it was generated for in its NAME -- the
// notebook writes `expt/Experiment5/{n_arms}arms_{n_outcomes}outcomes_{term}_presets.json`.
// We build the same name from the configured design, so N_BUTTONS/K_OUTCOMES pick
// the matching file and a mismatch shows up as a 404 rather than as histories
// silently mapped onto the wrong number of buttons or locations.
// TERMINATE picks the arm of the model the histories were generated under, using
// the same "Termination"/"noTermination" tag the notebook writes into the name.
const PRESETS_FILE = `rooms/${N_BUTTONS}arms_${K_OUTCOMES}outcomes_${TERMINATE ? "Termination" : "noTermination"}_presets.json`;

// Reject anything the display could not honestly show: a history referring to a
// button or a location this design does not have. `BUTTONS` and `OUTCOMES` are what
// actually get drawn, so they are the real bound, not N_BUTTONS/K_OUTCOMES.
function validatePresets(data) {
    if (!Array.isArray(data) || data.length === 0) {
        throw new Error(`${PRESETS_FILE} must be a non-empty list of histories`);
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
    data.forEach(function (history, r) {
        if (!Array.isArray(history) || history.length === 0) {
            throw new Error(`${PRESETS_FILE}: room ${r} is not a non-empty history`);
        }
        for (const entry of history) {
            const a = entry[0][0], o = entry[0][1], n = entry[1];
            if (!Number.isInteger(a) || !Number.isInteger(o) || !Number.isInteger(n) || n < 1) {
                throw new Error(`${PRESETS_FILE}: room ${r} has a malformed entry ${JSON.stringify(entry)}`);
            }
            if (a >= N_BUTTONS || o >= K_OUTCOMES || a < 0 || o < 0) {
                throw new Error(
                    `${PRESETS_FILE}: room ${r} uses action ${a} / outcome ${o}, outside ` +
                    `the ${N_BUTTONS}-button, ${K_OUTCOMES}-location design this file is named for`
                );
            }
            if (a > maxAction) maxAction = a;
            if (o > maxOutcome) maxOutcome = o;
        }
    });
    // not an error -- canonical histories legitimately leave later arms/outcomes
    // unobserved -- but worth seeing in the console when checking a new preset set
    console.log(
        `${PRESETS_FILE}: ${data.length} rooms, actions 0-${maxAction} of ${N_BUTTONS}, ` +
        `outcomes 0-${maxOutcome} of ${K_OUTCOMES}`
    );
}

// N_TRIALS is the room's press budget, and it sets two hard capacities:
//   - a counter cell draws exactly N_TRIALS token slots per button, so the
//     (button, location) with the most observations must fit inside it. Tokens
//     beyond that are silently NOT DRAWN -- the participant would press a button
//     and see nothing appear.
//   - with SAMPLE_COST = 1/(N_TRIALS+1), a room worth more than N_TRIALS presses
//     spends the whole coin before the participant chooses.
// Both are warnings rather than errors: a preset set can legitimately be paired
// with a larger N_TRIALS, and this tells you when it has to be.
function checkPresetCapacity() {
    const longest = Math.max.apply(null, ROOM_PRESETS.map(p => p.n_preset_presses));
    let fullestCell = 0;
    ROOM_PRESETS.forEach(function (p) {
        for (const b of BUTTONS) {
            for (const o of OUTCOMES) {
                if (p.presetCounts[b][o] > fullestCell) fullestCell = p.presetCounts[b][o];
            }
        }
    });
    // the participant's own presses can land on the already-fullest cell
    const worstCell = fullestCell + N_REMAINING_TRIALS;
    if (worstCell > N_TRIALS) {
        console.warn(
            `${PRESETS_FILE}: a counter cell may need ${worstCell} tokens ` +
            `(fullest preset cell ${fullestCell} + ${N_REMAINING_TRIALS} own press` +
            `${N_REMAINING_TRIALS === 1 ? "" : "es"}) but only ${N_TRIALS} slots are ` +
            `drawn. Raise N_TRIALS to at least ${worstCell} or tokens will go missing.`
        );
    }
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

// Fetch the presets file and derive N_ROOMS, the per-room button layouts and the
// per-room concretised presets. index.html builds its timeline inside this promise.
function loadPresets() {
    return fetch(PRESETS_FILE)
        .then(function (resp) {
            if (!resp.ok) throw new Error(`${PRESETS_FILE}: HTTP ${resp.status}`);
            return resp.json();
        })
        .then(function (data) {
            validatePresets(data);
            PRESETS = data;
            N_ROOMS = PRESETS.length;
            ROOM_BUTTON_ORDERS = Array.from({ length: N_ROOMS }, randomButtonOrder);
            ROOM_PRESETS = PRESETS.map(buildRoomPreset);
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
