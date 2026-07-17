//----------------------------------------------------------------------------//
// Attention checks.
// After every 2 rooms we show a short block of grids (ATTENTION_ROOMS_PER_BLOCK),
// each with a random red/blue token tally in every location, and ask which
// location has the most tokens of a given colour. The participant presses the
// ARROW KEY pointing to that location. No feedback is given.
//
// Once at least ATTENTION_MIN_CHECKS checks have been completed, if accuracy is
// below ATTENTION_PASS_FRACTION the participant is excluded (attention_failed is
// set true; index.html then skips the remaining task and shows an exclusion
// screen that redirects via REDIRECT_FAIL).
//----------------------------------------------------------------------------//
const ATTENTION_ROOMS_PER_BLOCK = 4;   // grids per attention-check block
const ATTENTION_MIN_CHECKS = ATTENTION_ROOMS_PER_BLOCK * 3;        // start evaluating once this many are done
const ATTENTION_PASS_FRACTION = 0.6;   // exclude below this running accuracy

// running tallies (read by index.html to gate the rest of the experiment)
let attention_total = 0;
let attention_correct = 0;
let attention_failed = false;

// arrow-key name -> outcome (jsPsych records the key as e.g. "ArrowUp")
const ARROW_TO_OUTCOME = {
    arrowup: "up", arrowright: "right", arrowdown: "down", arrowleft: "left"
};

function randTokenCount() {
    return Math.floor(Math.random() * (N_TRIALS + 1)); // 0..N_TRIALS tokens
}

// the outcome with a STRICT maximum count in cntObj, or null if the top is tied
function strictMaxOutcome(cntObj) {
    let best = null, bestVal = -1, tie = false;
    for (const o of OUTCOMES) {
        if (cntObj[o] > bestVal) { bestVal = cntObj[o]; best = o; tie = false; }
        else if (cntObj[o] === bestVal) { tie = true; }
    }
    return tie ? null : best;
}

// random red/blue token counts per location, regenerated until `color` has a
// unique maximum (so there is a single unambiguous correct answer).
function makeAttentionCounts(color) {
    let red, blue, correct;
    do {
        red = {}; blue = {};
        for (const o of OUTCOMES) { red[o] = randTokenCount(); blue[o] = randTokenCount(); }
        correct = strictMaxOutcome(color === "red" ? red : blue);
    } while (correct === null);
    return { red: red, blue: blue, correct: correct };
}

//----------------------------------------------------------------------------//
// One attention-check trial: a token grid + a "which location has the most
// <colour> tokens?" question answered with an arrow key. No feedback.
//----------------------------------------------------------------------------//
// the grid + "which location has the most <colour> tokens?" question HTML
function attentionStimulusHTML(color, red, blue) {
    return screenHTML({
        lines: [
            `Which location has the most
             <span style="color:rgb(${BTN_COLOR[color]}); font-weight:bold;">${color}</span> tokens?`,
            `Press the arrow key (&uarr; &rarr; &darr; &larr;) pointing to that location.`
        ],
        stage: `
            <div class="task-row" style="pointer-events:none;">
                ${roomCountersStaticHTML(red, blue)}
            </div>`
    });
}

function make_attention_trial() {
    // generate here (in the factory) so the values exist when the stimulus is
    // evaluated -- jsPsych processes parameters before on_start runs.
    const color = Math.random() < 0.5 ? "red" : "blue";
    const c = makeAttentionCounts(color);
    const red = c.red, blue = c.blue, correct = c.correct;
    return {
        type: jsPsychHtmlKeyboardResponse,
        choices: ["ArrowUp", "ArrowRight", "ArrowDown", "ArrowLeft"],
        stimulus: attentionStimulusHTML(color, red, blue),
        data: { task: "attention_check" },
        on_finish: function (data) {
            const chosen = ARROW_TO_OUTCOME[String(data.response).toLowerCase()] || null;
            const is_correct = chosen === correct;
            attention_total += 1;
            if (is_correct) attention_correct += 1;

            data.attention_color = color;
            data.attention_counts = { red: red, blue: blue };
            data.attention_correct_outcome = correct;
            data.attention_chosen_outcome = chosen;
            data.attention_is_correct = is_correct;
            data.attention_total = attention_total;
            data.attention_correct_count = attention_correct;
            data.attention_accuracy = attention_correct / attention_total;

            // once enough checks are done, exclude if running accuracy is too low
            if (attention_total >= ATTENTION_MIN_CHECKS &&
                (attention_correct / attention_total) < ATTENTION_PASS_FRACTION) {
                attention_failed = true;
            }
        }
    };
}

// a block of ATTENTION_ROOMS_PER_BLOCK checks; each is skipped once excluded, so
// the block halts as soon as the failure criterion is met.
function make_attention_block() {
    const block = [];
    for (let i = 0; i < ATTENTION_ROOMS_PER_BLOCK; i++) {
        block.push({
            timeline: [make_attention_trial()],
            conditional_function: function () { return !attention_failed; }
        });
    }
    return block;
}

// exclusion end-screen, shown only when the attention criterion was failed.
function make_attention_excluded_trial() {
    return {
        type: jsPsychHtmlButtonResponse,
        choices: ["Finish"],
        stimulus: screenHTML({
            title: `The experiment has ended`,
            lines: [
                `Unfortunately, based on the attention checks, you do not meet the criteria to continue with the task.`,
                `Thank you for your time.`,
                `Click below to return to Prolific.`
            ]
        }),
        data: { task: "attention_excluded" }
    };
}

//----------------------------------------------------------------------------//
// Attention-check introduction + a short practice WITH feedback (shown after the
// comprehension check, before the main task). The practice does NOT count towards
// the exclusion tally.
//----------------------------------------------------------------------------//
const ATTENTION_PRACTICE_ITEMS = 3;

// one practice item = the question (no scoring) + a feedback screen.
function make_attention_practice_item() {
    const color = Math.random() < 0.5 ? "red" : "blue";
    const c = makeAttentionCounts(color);
    const red = c.red, blue = c.blue, correct = c.correct;
    let chosen = null, is_correct = false;

    const question = {
        type: jsPsychHtmlKeyboardResponse,
        choices: ["ArrowUp", "ArrowRight", "ArrowDown", "ArrowLeft"],
        stimulus: attentionStimulusHTML(color, red, blue),
        data: { task: "attention_practice" },
        on_finish: function (data) {
            chosen = ARROW_TO_OUTCOME[String(data.response).toLowerCase()] || null;
            is_correct = chosen === correct;
            data.attention_color = color;
            data.attention_correct_outcome = correct;
            data.attention_chosen_outcome = chosen;
            data.attention_is_correct = is_correct;
        }
    };
    const feedback = {
        type: jsPsychHtmlButtonResponse,
        choices: ["Continue"],
        stimulus: function () {
            const head = is_correct
                ? `<span style="color:#2ca02c;">Correct!</span>`
                : `<span style="color:#c0392b;">Incorrect</span>`;
            return screenHTML({
                title: head,
                lines: [
                    `The most <span style="color:rgb(${BTN_COLOR[color]}); font-weight:bold;">${color}</span>
                     tokens were in the <strong>${correct}</strong> location.`
                ],
                stage: `
                    <div class="task-row" style="pointer-events:none;">
                        ${roomCountersStaticHTML(red, blue)}
                    </div>`
            });
        },
        data: { task: "attention_practice_feedback" }
    };
    return [question, feedback];
}

// full intro + practice timeline (returns an array of trials).
function make_attention_intro_and_practice() {
    const tl = [];

    tl.push({
        type: jsPsychInstructions,
        show_clickable_nav: true,
        button_label_previous: "Previous",
        button_label_next: "Next",
        data: { task: "attention_intro" },
        pages: [
            screenHTML({
                title: `Attention checks`,
                lines: [
                    `Every so often during the task, we'll check that you're still paying attention.`,
                    `You'll see a room with red and blue tokens, and we'll ask: <strong>which location has the most tokens of a particular colour?</strong>`,
                    `Answer by pressing the <strong>arrow key</strong> (&uarr; &rarr; &darr; &larr;) pointing to that location.`,
                    `Let's <strong>practise</strong> a couple, with feedback.`
                ]
            })
        ]
    });

    for (let i = 0; i < ATTENTION_PRACTICE_ITEMS; i++) {
        tl.push.apply(tl, make_attention_practice_item());
    }

    tl.push({
        type: jsPsychInstructions,
        show_clickable_nav: true,
        button_label_previous: "Previous",
        button_label_next: "Next",
        data: { task: "attention_intro_end" },
        pages: [
            screenHTML({
                title: `Ready?`,
                lines: [
                    `In the real attention checks you <strong>won't</strong> be told whether you were right.`,
                    `Please answer them carefully &mdash; if too many are missed, the study may end early.`
                ]
            })
        ]
    });

    return tl;
}
