//----------------------------------------------------------------------------//
// Shared building blocks
//----------------------------------------------------------------------------//
// The two circle buttons, vertically stacked to the right of the room.
function buttonStackHTML() {
    return `
        <div class="button-stack">
            <div class="cbtn blue" id="btn-blue"></div>
            <div class="cbtn red" id="btn-red"></div>
        </div>`;
}

// The "done sampling" tick button, shown to the left of the room.
function checkButtonHTML() {
    return `
        <div class="check-stack">
            <div class="checkbtn" id="btn-check"><img src="img/Check.png" alt="Done sampling"></div>
            <div class="check-label">Done<br>sampling</div>
        </div>`;
}

// Right-hand belief panel: colorbar legend (overlay) or two belief grids (separate).
function beliefPanelHTML() {
    // counters + overlay draw into the main grid; counters has no side panel.
    if (BELIEF_DISPLAY === "counters") {
        return ``;
    }
    if (BELIEF_DISPLAY === "overlay") {
        return `
            <div class="belief-stack">
                ${colorbarLegendHTML("red")}
                ${colorbarLegendHTML("blue")}
            </div>`;
    }
    return `
        <div class="belief-stack">
            ${beliefBlockHTML("blue")}
            ${beliefBlockHTML("red")}
        </div>`;
}

// Redraw the belief display (and the sample-counter titles) from current counts.
function refreshBeliefs(highlight) {
    if (BELIEF_DISPLAY === "counters") {
        renderMainCounters(highlight);
    } else if (BELIEF_DISPLAY === "overlay") {
        renderMainBeliefOverlay();
    } else {
        renderBeliefGrid("red", document.getElementById("belief-red"));
        renderBeliefGrid("blue", document.getElementById("belief-blue"));
    }
    updateBeliefLabels();
}

// Wire click handlers onto the buttons; `onPress(button, rt)` handles a colour
// choice. If a tick button is present and `onCheck` is given, `onCheck(rt)` fires
// when it is clicked. rt is measured from when the buttons become available.
function wireButtons(onPress, onCheck) {
    const btnRed = document.getElementById("btn-red");
    const btnBlue = document.getElementById("btn-blue");
    const btnCheck = document.getElementById("btn-check");
    const t0 = performance.now();

    function disableAll() {
        btnRed.classList.add("disabled");
        btnBlue.classList.add("disabled");
        if (btnCheck) btnCheck.classList.add("disabled");
    }

    function handle(button) {
        const rt = Math.round(performance.now() - t0);
        disableAll();
        // hide the unselected colour button and the whole tick (button + label)
        (button === "red" ? btnBlue : btnRed).classList.add("hidden");
        if (btnCheck) (btnCheck.closest(".check-stack") || btnCheck).classList.add("hidden");
        onPress(button, rt);
    }
    btnRed.addEventListener("click", () => handle("red"));
    btnBlue.addEventListener("click", () => handle("blue"));

    if (btnCheck && onCheck) {
        btnCheck.addEventListener("click", function () {
            const rt = Math.round(performance.now() - t0);
            disableAll();
            onCheck(rt);
        });
    }
}

//----------------------------------------------------------------------------//
// Room intro: reset beliefs and draw fresh hidden transition functions.
//----------------------------------------------------------------------------//
function make_room_intro(room_num) {
    return {
        type: jsPsychHtmlKeyboardResponse,
        stimulus: `<div style="text-align:center">
            <h2>Room ${room_num} of ${N_ROOMS}</h2>
            <h3>Sample the two buttons over ${N_TRIALS} trials to learn where each one takes you.</h3>
            <h3>Press any key to begin.</h3>
        </div>`,
        data: { task: "room_intro", room_num: room_num },
        on_start: function () {
            for (const b of BUTTONS) {
                for (const o of OUTCOMES) counts[b][o] = 0;
            }
            // draw this room's hidden transition functions from Dirichlet(ALPHA)
            sampleTrueT();
            // allow sampling again in this new room
            sampling_ended = false;
        },
        on_finish: function (data) {
            // log this room's (hidden) generative transition functions for analysis
            data.true_T = JSON.parse(JSON.stringify(TRUE_T));
        }
    };
}

//----------------------------------------------------------------------------//
// One sampling trial: click a circle, the agent moves to the sampled outcome,
// and the belief display re-shades once the agent has arrived.
//----------------------------------------------------------------------------//
function make_trial(room_num, trial_num, opts) {
    opts = opts || {};
    const practice = opts.practice === true;
    const taskName = practice ? "practice_sample" : "sample";
    // const hint = practice
    //     ? `<h4 style="color:#c99700; font-weight:normal; margin-top:6px;">Practice: test the buttons as you like, then click the tick when you feel you've learned enough.</h4>`
    //     : ``;

    return {
        type: jsPsychHtmlKeyboardResponse,
        choices: "NO_KEYS",
        stimulus: `
            <div class="task-row">
                ${checkButtonHTML()}
                ${initialize_agent()}
                ${buttonStackHTML()}
                ${beliefPanelHTML()}
            </div>
            <div class="prompt">
                <h4 class="trial-counter">Trial ${trial_num} of ${N_TRIALS}</h4>
                <h3>Click a coloured button to move, or the tick to finish sampling.</h3>
            </div>`,
        data: {
            task: taskName,
            room_num: room_num,
            trial_num: trial_num
        },
        on_start: function () {
            // reset agent to the central cell at the start of each trial
            agent_topPos = topPos0;
            agent_leftPos = leftPos0;
        },
        on_load: function () {
            refreshBeliefs();

            wireButtons(function (button, rt) {
                // belief state the choice was based on (before this observation)
                const counts_pre = countsSnapshot();
                const posteriors_pre = posteriorSnapshot();

                const outcome = sampleCategorical(TRUE_T[button]);
                counts[button][outcome] += 1;

                // the agent moves first; the posterior/colours only update once it
                // has actually reached the outcome (after the move animation).
                moveAgent(outcome);

                const trial_data = {
                    chosen_button: button,
                    outcome: outcome,
                    rt: rt,
                    ended_early: false,
                    counts: counts_pre,           // transition counts at decision time
                    posterior_means: posteriors_pre,
                    counts_post: countsSnapshot() // counts after this observation
                };

                setTimeout(function () {
                    // agent has arrived -> reveal the updated belief; in counters
                    // mode, pop in the token just placed for this observation
                    refreshBeliefs({ button: button, outcome: outcome });
                    setTimeout(() => jsPsych.finishTrial(trial_data), 900);
                }, MOVE_MS);
            }, function (rt) {
                // tick button: end sampling early and skip to the coin phase
                sampling_ended = true;
                jsPsych.finishTrial({
                    task: taskName,
                    ended_early: true,
                    rt: rt,
                    counts: countsSnapshot(),
                    posterior_means: posteriorSnapshot()
                });
            });
        }
    };
}

//----------------------------------------------------------------------------//
// Gold collection: a coin appears at a random reachable cell; the participant
// picks a button to try to reach it.
//   SHOW_GOLD_OUTCOME true  -> agent moves per the transition function, feedback shown.
//   SHOW_GOLD_OUTCOME false -> no outcome shown; move straight on to the next room.
//----------------------------------------------------------------------------//
function make_gold_trial(room_num) {
    return {
        type: jsPsychHtmlKeyboardResponse,
        choices: "NO_KEYS",
        stimulus: `
            <div class="task-row">
                ${initialize_agent_gold()}
                ${buttonStackHTML()}
                ${beliefPanelHTML()}
            </div>
            <div class="prompt">
                <h4 class="trial-counter">Gold time!</h4>
                <h3>A gold coin appeared. Click the button most likely to reach it.</h3>
                <h2 id="gold-feedback"></h2>
            </div>`,
        data: { task: "gold", room_num: room_num },
        on_start: function () {
            agent_topPos = topPos0;
            agent_leftPos = leftPos0;
        },
        on_load: function () {
            // gold appears at a random reachable (cardinal) cell
            const goldOutcome = OUTCOMES[Math.floor(Math.random() * OUTCOMES.length)];
            placeGold(goldOutcome);
            refreshBeliefs();

            wireButtons(function (button, rt) {
                const trial_data = {
                    chosen_button: button,
                    gold_outcome: goldOutcome,
                    rt: rt,
                    counts: countsSnapshot(),                 // final transition counts for the room
                    posterior_means: posteriorSnapshot()      // belief the choice was based on
                };

                if (!SHOW_GOLD_OUTCOME) {
                    // do not reveal the outcome; move straight on to the next room
                    trial_data.outcome_shown = false;
                    setTimeout(() => jsPsych.finishTrial(trial_data), 400);
                    return;
                }

                // reveal: the agent moves according to the transition function
                const outcome = sampleCategorical(TRUE_T[button]);
                const success = outcome === goldOutcome;
                trial_data.outcome_shown = true;
                trial_data.outcome = outcome;
                trial_data.success = success;
                if (success) collected_gold += 1;
                trial_data.collected_gold = collected_gold;

                moveAgent(outcome);

                setTimeout(function () {
                    // show whether the gold was obtained, once the agent has arrived
                    const fb = document.getElementById("gold-feedback");
                    if (success) {
                        fb.textContent = "You got the gold!";
                        fb.style.color = "green";
                    } else {
                        fb.textContent = "Missed it...";
                        fb.style.color = "red";
                    }


                    // save data so far
                    var ppt_data = jsPsych.data.get().json();
                    send_incomplete(id, ppt_data);

                    setTimeout(() => jsPsych.finishTrial(trial_data), 1400);
                }, MOVE_MS);
            });
        }
    };
}
