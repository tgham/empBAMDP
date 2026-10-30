//----------------------------------------------------------------------------//
// Comprehension check.
//
// A multiple-choice quiz shown after the instructions. The participant must get
// at least 2/3 of the questions correct. If they don't, they review the
// instructions and try once more; a second failure ends the experiment (with a
// redirect to Prolific). Grading + pass/fail bookkeeping happen in the quiz
// trial's on_finish, and the loop / conditional logic lives in index.html
// (globals: quiz_passed, quiz_attempts).
//----------------------------------------------------------------------------//

// Each question: prompt, the correct option, and the distractors. `costOnly`,
// `tickOnly`, `goldOnly` and `controlOnly` mark questions that only make sense
// under some settings -- see the filter below. Distractors that would mention
// gold are swapped for a neutral alternative when OBJECTIVE is "control", since
// those participants never hear about gold.
const ROOM_SET = N_AFC === 2 ? "pair" : "set";
const ALL_QUIZ_QUESTIONS = [
    {
        name: "buttons",
        prompt: "In each room, what happens when a button is pressed?",
        correct: "It leads to one of the locations, tending to reach some locations more often than others.",
        distractors: [
            "It always leads to the same single, fixed location.",
            OBJECTIVE === "gold"
                ? "It picks up a gold coin directly."
                : "It reaches every location equally often."
        ]
    },
    {
        name: "blue_up",
        prompt: "Suppose the blue button is pressed and it leads to the upward location. What can you infer about the upward location?",
        correct: "It's evidence that blue is fairly likely to lead upward, but not a certainty.",
        distractors: [
            "Blue will now always lead upward.",
            OBJECTIVE === "gold"
                ? "The gold coin must be in the upward location."
                : "The other buttons will now also lead upward."
        ]
    },
    {
        name: "red_left",
        prompt: "Suppose the red button is pressed and it leads to the leftward location. What can you infer about the up, down and rightward locations?",
        correct: "It's evidence that red is a little less likely to lead to those locations (though they're still possible).",
        distractors: [
            "Those locations are impossible to reach with the red button.",
            "The blue button will definitely lead to one of those locations."
        ]
    },
    {
        name: "relationship",
        prompt: `What is the relationship between the buttons in each ${ROOM_SET} of rooms?`,
        correct: "They all control movement in that room, and the locations they reach are independent of each other. The buttons in one room are also unrelated to the buttons in the other room(s).",
        distractors: [
            "The buttons in one room are always more reliable than those in the other room(s).",
            "A button in one room works the same way as the button of the same colour in the other room(s)."
        ]
    },
    {
        name: "heatmap",
        prompt: "What information does a button's tokens show?",
        correct: "The number of times each location has been reached by that button in that room during testing.",
        distractors: [
            OBJECTIVE === "gold"
                ? "How likely it is that a gold coin will appear in each location."
                : "The order in which the locations were reached.",
            "How many points have been earned by pressing that button."
        ]
    },
    {
        name: "new_room",
        prompt: "Suppose one of the buttons hasn't been tried yet. What can you assume about this button, based on your current knowledge?",
        correct: "Since it hasn't been tried, it's reasonable to assume it has a similar chance of reaching each location.",
        distractors: [
            "Since it hasn't been tried yet, it is almost definitely a reliable button.",
            "Since it hasn't been tried yet, it is almost definitely an unreliable button."
        ]
    },
    {
        name: "auto_presses",
        prompt: `When you encounter a new ${ROOM_SET} of rooms, which buttons are automatically pressed?`,
        correct: "The number and colour of the button presses in each room vary from trial to trial.",
        distractors: [
            "Each button of each room is pressed evenly.",
            "The most reliable buttons are pressed."
        ]
    },
    {
        name: "room_choice",
        prompt: "Which room should you select?",
        correct: OBJECTIVE === "gold"
            ? "The room in which I think I have the best chance of reaching the gold coin."
            : "The room in which I feel the strongest sense of control.",
        distractors: [
            "The room in which the buttons have been pressed the most times.",
            OBJECTIVE === "gold"
                ? "The room with the most tokens in any single location."
                : "The room with the fewest tokens."
        ]
    },
    {
        name: "control",
        controlOnly: true,
        prompt: "Which room provides the strongest sense of control?",
        correct: "It is open to interpretation - I should answer as honestly as possible.",
        distractors: [
            "The room with more buttons.",
            "The room on the left side of the screen."
        ]
    },
    {
        name: "cost",
        goldOnly: true,
        costOnly: true,
        prompt: "What is the cost of testing a button?",
        correct: "Testing a button reduces the size, and therefore value, of the gold coin that I can possibly collect in the current room.",
        distractors: [
            "Testing a button reduces the number of rooms I can visit.",
            "None - testing a button is always beneficial."
        ]
    },
    {
        name: "tick",
        goldOnly: true,
        tickOnly: true,
        prompt: "What happens if you press the orange tick button?",
        correct: "I stop testing early and move straight on to the gold collection phase.",
        distractors: [
            "I collect a gold coin.",
            "I reset the room and start it again."
        ]
    },
    {
        name: "coin_size",
        goldOnly: true,
        costOnly: true,
        prompt: "In the gold collection phase, what is the value of the gold coin that can possibly be collected?",
        correct: "The value depends on the size of the coin, which is determined by the number of buttons pressed.",
        distractors: [
            "The gold coin is always the same size, so the value is always the same.",
            "The value depends on the number of gold coins I have already collected."
        ]
    },
    {
        name: "coin",
        goldOnly: true,
        prompt: "In the gold collection phase, which button should you choose?",
        correct: "The button that I believe will most likely lead to the location with the coin.",
        distractors: [
            "The button with the highest number of combined tokens.",
            "The button with the most tokens in any single location."
        ]
    },
    {
        name: "coin_correct",
        goldOnly: true,
        prompt: "In the gold collection phase, is there a button that guarantees leading to the location with the coin?",
        correct: "Not necessarily - sometimes no button reliably leads to the location with the coin.",
        distractors: [
            "Yes - there is always one button that certainly leads to the location with the coin.",
            "Yes - the button with the most tokens certainly leads to the location with the coin."
        ]
    },
];

// With SAMPLE_COST = 0 nothing depletes the coin, so the two cost questions have
// no right answer for a participant who understood the task. Same for the tick
// question when TERMINATE is off and there is no tick to press, and for the gold
// questions under the "control" objective (no gold phase). Drop them rather than
// grade people on a rule the instructions (correctly) never taught.
const QUIZ_QUESTIONS = ALL_QUIZ_QUESTIONS.filter(
    q => (SAMPLE_COST > 0 || !q.costOnly) && (TERMINATE || !q.tickOnly) &&
         (OBJECTIVE === "gold" || !q.goldOnly) && (OBJECTIVE === "control" || !q.controlOnly)
);

const QUIZ_PASS_FRACTION = 2/3;
const QUIZ_MAX_ATTEMPTS = 2;

//----------------------------------------------------------------------------//
// The quiz trial (all questions on one page, options shuffled per question).
//----------------------------------------------------------------------------//
function make_comprehension_quiz() {
    const questions = QUIZ_QUESTIONS.map(function (q) {
        const options = jsPsych.randomization.shuffle([q.correct].concat(q.distractors));
        return { prompt: q.prompt, name: q.name, options: options, required: true };
    });

    return {
        type: jsPsychSurveyMultiChoice,
        preamble: `<h2>Comprehension check</h2>
                   <p style="max-width:680px; margin:14px auto;">Please answer these questions about the task.
                   You need at least <strong>${Math.ceil(QUIZ_PASS_FRACTION * QUIZ_QUESTIONS.length)} of
                   ${QUIZ_QUESTIONS.length}</strong> correct to continue.</p>`,
        questions: questions,
        data: { task: "comprehension" },
        on_finish: function (data) {
            let correct = 0;
            QUIZ_QUESTIONS.forEach(function (q) {
                if (data.response[q.name] === q.correct) correct++;
            });
            quiz_attempts += 1;
            quiz_passed = (correct / QUIZ_QUESTIONS.length) >= QUIZ_PASS_FRACTION;
            data.quiz_correct = correct;
            data.quiz_total = QUIZ_QUESTIONS.length;
            data.quiz_passed = quiz_passed;
            data.quiz_attempt = quiz_attempts;
        }
    };
}

//----------------------------------------------------------------------------//
// Feedback after the quiz (message depends on pass / attempt number).
//----------------------------------------------------------------------------//
function make_quiz_result_trial() {
    return {
        type: jsPsychHtmlButtonResponse,
        choices: ["Continue"],
        data: { task: "quiz_result" },
        stimulus: function () {
            const last = jsPsych.data.get().filter({ task: "comprehension" }).last(1).values()[0];
            const score = `You got <strong>${last.quiz_correct} of ${last.quiz_total}</strong> correct.`;
            if (quiz_passed) {
                return `<h2>Nice work &mdash; you passed!</h2>
                        <p style="max-width:640px; margin:14px auto;">${score}</p>`;
            }
            if (quiz_attempts < QUIZ_MAX_ATTEMPTS) {
                return `<h2>Not quite</h2>
                        <p style="max-width:640px; margin:14px auto;">${score}</p>
                        <p style="max-width:640px; margin:14px auto;">Let's go through the instructions again, then
                        you can retake the check.</p>`;
            }
            return `<h2>Thanks for taking part</h2>
                    <p style="max-width:640px; margin:14px auto;">${score}</p>
                    <p style="max-width:640px; margin:14px auto;">Unfortunately, you didn't score enough on the comprehension check to continue. 
                    <p style="max-width:640px; margin:14px auto;">You will now be redirected to Prolific, where you will be reimbused for your time.</p>`;
        }
    };
}
