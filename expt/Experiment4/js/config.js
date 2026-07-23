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
const N_BUTTONS = 2;
const K_OUTCOMES = 4;
const N_TRIALS = 8;   // sampling trials per room
const N_ROOMS = 25;    // number of rooms (fresh transition functions each)

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
const BUTTONS = ["red", "blue"];
const BTN_COLOR = { red: "220,40,40", blue: "40,90,220" }; // rgb triples for rgba() shading

// The two buttons are shown one above the other, but offset diagonally, and the
// colour->position mapping is randomised once per participant. This prevents a
// spatial bias (e.g. reading "blue = up, red = down" from a fixed vertical stack).
// BUTTON_ORDER = [upper colour, lower colour]; the lower button sits to the right.
// ROOM_BUTTON_ORDERS is pre-generated once per experiment so each room has a
// consistent, predetermined layout that persists across all trials in that room.
let BUTTON_ORDER = Math.random() < 0.5 ? ["blue", "red"] : ["red", "blue"];
const ROOM_BUTTON_ORDERS = Array.from({ length: N_ROOMS }, () => 
    Math.random() < 0.5 ? ["blue", "red"] : ["red", "blue"]
);

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

//----------------------------------------------------------------------------//
// Hidden true transition distributions (categorical over OUTCOMES).
// These are unknown to the participant; each press samples one outcome from them.
// They are drawn fresh for each room from a symmetric Dirichlet(ALPHA) prior --
// the SAME prior the participant's posterior assumes -- via sampleTrueT() (see
// render.js), called at the start of each block.
//----------------------------------------------------------------------------//
let TRUE_T = {
    red:  { up: 0.25, right: 0.25, down: 0.25, left: 0.25 },
    blue: { up: 0.25, right: 0.25, down: 0.25, left: 0.25 }
};

// Which context (1 or 2) each button's TRUE_T was drawn from in the current room.
// Set by sampleTrueT(); null for every button when CONTEXTUAL is false.
let BUTTON_CTX = { red: null, blue: null };

//----------------------------------------------------------------------------//
// Observation counts. Persist across trials so beliefs accumulate over T trials.
//----------------------------------------------------------------------------//
let counts = {
    red:  { up: 0, right: 0, down: 0, left: 0 },
    blue: { up: 0, right: 0, down: 0, left: 0 }
};

// running total of gold coins collected across rooms
let collected_gold = 0;

// set true when the participant ends sampling early (via the tick button); reset
// at the start of each room so the remaining sampling trials are skipped.
let sampling_ended = false;
