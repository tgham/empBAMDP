//----------------------------------------------------------------------------//
// Experiment parameters
//----------------------------------------------------------------------------//
// N=2 buttons, K=4 outcomes (cardinal directions), T=6 trials, Dirichlet alpha=1
const N_BUTTONS = 2;
const K_OUTCOMES = 4;
const N_TRIALS = 8;   // sampling trials per room
const N_ROOMS = 5;    // number of rooms (fresh transition functions each)
const ALPHA = 1;

// After the sampling trials, a gold coin appears at a random reachable cell and
// the participant picks a button to try to reach it.
//   true  -> reveal the outcome: the agent moves per the transition function and
//            we show whether the gold was obtained.
//   false -> do not reveal the outcome; move straight on to the next room.
const SHOW_GOLD_OUTCOME = true;

// Prolific redirect links (fill in later). If left blank, no redirect happens.
const REDIRECT_COMPLETE = ""; // shown after finishing the experiment
const REDIRECT_FAIL = "";     // shown after failing the comprehension check twice

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

//----------------------------------------------------------------------------//
// Belief display mode:
//   "overlay"  -> both buttons' posteriors shown in the main grid, each reachable
//                 cell split diagonally (red upper-right triangle, blue lower-left).
//   "separate" -> a dedicated belief grid per button, to the right of the room.
//----------------------------------------------------------------------------//
// const BELIEF_DISPLAY = "overlay";
const BELIEF_DISPLAY = "separate";

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
