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
// N=2 buttons, K=4 outcomes (cardinal directions), T=6 trials, Dirichlet alpha=1
const N_BUTTONS = 3;
const K_OUTCOMES = 4;
const N_TRIALS = 8;   // sampling trials per room
const N_ROOMS = 30;    // number of rooms (fresh transition functions each)

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

function randomButtonOrder() {
    const arr = BUTTONS.slice();
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
}
let BUTTON_ORDER = randomButtonOrder();
const ROOM_BUTTON_ORDERS = Array.from({ length: N_ROOMS }, () => randomButtonOrder());

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
