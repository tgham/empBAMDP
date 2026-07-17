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

// Each question: prompt, the correct option, and the distractors.
const QUIZ_QUESTIONS = [
    {
        name: "buttons",
        prompt: "In each room, what do the two buttons let you do?",
        correct: "Pressing a button moves me to one of the locations, tending to reach some locations more often than others.",
        distractors: [
            "Pressing a button always moves me to the same single, fixed location.",
            "The buttons let me pick up gold coins directly."
        ]
    },
    {
        name: "blue_up",
        prompt: "Suppose I press the blue button and it takes me to the upward location. What can I infer about the upward location?",
        correct: "It's evidence that blue is fairly likely to lead upward, but not a certainty.",
        distractors: [
            "Blue will now always lead upward.",
            "The gold coin must be in the upward location."
        ]
    },
    {
        name: "red_left",
        prompt: "Suppose I press the red button and it takes me to the leftward location. What can I infer about the up, down and rightward locations?",
        correct: "It's evidence that red is a little less likely to lead to those locations (though they're still possible).",
        distractors: [
            "Those locations are impossible to reach with the red button.",
            "The blue button will definitely lead me to one of those locations."
        ]
    },
    {
        name: "relationship",
        prompt: "What is the relationship between the two buttons?",
        correct: "They both control movement in the same room, but the locations they reach are independent of each other.",
        distractors: [
            "They always reach the same locations as each other.",
            "They operate in two completely separate rooms."
        ]
    },
    {
        name: "heatmap",
        prompt: "What information does a button's tokens show?",
        correct: "The number of times each location has been reached by that button during testing.",
        distractors: [
            "How likely it is that a gold coin will appear in each location.",
            "How many points I have earned by pressing that button."
        ]
    },
    {
        name: "tick",
        prompt: "What happens if you press the green tick button?",
        correct: "I stop sampling early and move straight on to the coin selection.",
        distractors: [
            "I collect a gold coin.",
            "I reset the room and start it again."
        ]
    },
    {
        name: "coin",
        prompt: "In the coin selection phase, which button should I choose?",
        correct: "The button that I believe will most likely lead to the location with the coin.",
        distractors: [
            "The button with the highest number of combined tokens.",
            "The button with the most tokens in any single location."
        ]
    },
    {
        name: "coin_correct",
        prompt: "In the coin selection phase, is there a button that guarantees leading to the location with the coin?",
        correct: "Not necessarily - sometimes neither button reliably leads to the location with the coin.",
        distractors: [
            "Yes - there is always one button that certainly leads to the location with the coin.",
            "Yes - the button with the most tokens certainly leads to the location with the coin."
        ]
    },
];

const QUIZ_PASS_FRACTION = 2 / 3;
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
