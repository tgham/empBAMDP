//----------------------------------------------------------------------------//
// Shared building blocks
//----------------------------------------------------------------------------//
// The two circle buttons, to the right of the room. Stacked upper/lower but
// offset diagonally, with the colour->position mapping randomised per participant
// so position isn't associated with a colour. buttonOrder parameter allows each
// room to capture its own layout independently.
function buttonStackHTML(opts) {
    opts = opts || {};
    const label_on = opts.label_on === true;
    const buttonOrder = opts.buttonOrder || BUTTON_ORDER;
    const upper = buttonOrder[0], lower = buttonOrder[1];
    function labelFor(color) {
        return color === "red" ? "Red button" : "Blue button";
    }
    return `
        <div class="button-stack">
            <div class="button-stack-item">
                <div class="button-label${label_on ? "" : " hidden"}">${label_on ? labelFor(upper) : ""}</div>
                <div class="cbtn ${upper}" id="btn-${upper}"></div>
            </div>
            <div class="button-stack-item cbtn-lower">
                <div class="button-label${label_on ? "" : " hidden"}">${label_on ? labelFor(lower) : ""}</div>
                <div class="cbtn ${lower}" id="btn-${lower}"></div>
            </div>
        </div>`;
}

// The "done sampling" tick button, shown to the left of the room. With
// { placeholder: true } it renders the same element but invisible (keeping its
// layout footprint) so grids without a tick don't shift horizontally.
function checkButtonHTML(opts) {
    const placeholder = opts && opts.placeholder;
    const tick_label = opts && opts.tick_label;
    return `
        <div class="check-stack${placeholder ? " hidden" : ""}"${placeholder ? " aria-hidden=\"true\"" : ""}>
            ${tick_label ? `<div class="check-label">Tick button</div>` : ""}
            <div class="checkbtn" id="btn-check"><img src="img/Check.png" alt="Done testing"></div>
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

// How many of the room's presses are still to come, e.g. "8 choices remaining".
function choicesRemainingText(remaining) {
    return `${remaining} choice${remaining === 1 ? "" : "s"} remaining`;
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
        stimulus: screenHTML({
            title: `Room ${room_num} of ${N_ROOMS}`,
            lines: [
                `Test out the two buttons over up to ${N_TRIALS} choices to learn where each one takes you.`,
                `Press any key to begin.`
            ]
        }),
        data: { task: "room_intro", room_num: room_num },
        on_start: function () {
            for (const b of BUTTONS) {
                for (const o of OUTCOMES) counts[b][o] = 0;
            }
            // draw this room's hidden transition functions from the Dirichlet prior
            sampleTrueT();
            // allow sampling again in this new room
            sampling_ended = false;
        },
        on_finish: function (data) {
            // log this room's (hidden) generative transition functions for analysis,
            // and which context each was drawn from (null unless CONTEXTUAL)
            data.true_T = JSON.parse(JSON.stringify(TRUE_T));
            data.button_ctx = JSON.parse(JSON.stringify(BUTTON_CTX));
        }
    };
}

//----------------------------------------------------------------------------//
// One room's whole sampling phase, as a SINGLE jsPsych trial. Keeping every
// press inside one trial means the grid DOM is built once and never torn down
// between presses, so there is no inter-trial flicker. Each press is still
// recorded as its own data row (task "sample"/"practice_sample"), pushed
// directly onto the data collection so the trial-by-trial data shape is
// unchanged from the old one-trial-per-press version.
//
// The tick ("done testing") ends the phase early. The trial ends when the
// participant has used all N_TRIALS presses or clicks the tick.
//----------------------------------------------------------------------------//
function make_room_sampling(room_num, opts) {
    opts = opts || {};
    const practice = opts.practice === true;
    const taskName = practice ? "practice_sample" : "sample";
    const buttonOrder = opts.buttonOrder || BUTTON_ORDER;

    // Manually-pushed rows don't get the session-level fields that jsPsych
    // auto-applies to rows it writes itself, so stamp them on here to match.
    function stampSession(row) {
        row.subject_id = subject_id;
        row.study_id = study_id;
        row.session_id = session_id;
        row.belief_display = BELIEF_DISPLAY;
        row.alpha = ALPHA;
        row.contextual = CONTEXTUAL;
        row.alpha_ctx1 = ALPHA_CTX1;
        row.alpha_ctx2 = ALPHA_CTX2;
        row.button_upper = buttonOrder[0];
        row.button_lower = buttonOrder[1];
        row.trial_type = "html-keyboard-response";
        return row;
    }

    return {
        type: jsPsychHtmlKeyboardResponse,
        choices: "NO_KEYS",
        stimulus: screenHTML({
            lines: [
                `<span class="trial-counter">${choicesRemainingText(N_TRIALS)}</span>`,
                `Click a button to move, or the tick to finish testing.`
            ],
            stage: `
                <div class="task-row">
                    ${checkButtonHTML()}
                    ${initialize_agent()}
                    ${buttonStackHTML({ buttonOrder: buttonOrder })}
                    ${beliefPanelHTML()}
                </div>`
        }),
        data: { task: "room_sampling", room_num: room_num, practice: practice },
        on_start: function () {
            // agent starts in the central cell
            agent_topPos = topPos0;
            agent_leftPos = leftPos0;
        },
        on_load: function () {
            refreshBeliefs();

            let trial_num = 1;   // 1..N_TRIALS; the press currently being made
            let ended = false;

            function finishRoom() {
                ended = true;
                jsPsych.finishTrial({
                    task: "room_sampling",
                    room_num: room_num,
                    practice: practice,
                    n_presses: trial_num - 1
                });
            }

            // Re-render the controls (fresh, so no stale click handlers accumulate)
            // and update the trial counter, then re-arm the buttons for the next
            // press. The agent grid + counters layer are left in place (persist).
            function armPress() {
                const row = document.querySelector(".task-row");
                const bs = row.querySelector(".button-stack");
                if (bs) bs.outerHTML = buttonStackHTML({ buttonOrder: buttonOrder });
                const cs = row.querySelector(".check-stack");
                if (cs) cs.outerHTML = checkButtonHTML();
                const counter = document.querySelector(".trial-counter");
                if (counter) counter.textContent = choicesRemainingText(N_TRIALS - trial_num + 1);
                wireButtons(onPress, onCheck);
            }

            function onPress(button, rt) {
                // belief state the choice was based on (before this observation)
                const counts_pre = countsSnapshot();
                const posteriors_pre = posteriorSnapshot();

                const outcome = sampleCategorical(TRUE_T[button]);
                counts[button][outcome] += 1;

                // the agent moves first; beliefs update only once it has arrived
                moveAgent(outcome);

                jsPsych.data.get().push(stampSession({
                    task: taskName,
                    room_num: room_num,
                    trial_num: trial_num,
                    chosen_button: button,
                    outcome: outcome,
                    rt: rt,
                    ended_early: false,
                    counts: counts_pre,            // transition counts at decision time
                    posterior_means: posteriors_pre,
                    counts_post: countsSnapshot()  // counts after this observation
                }));

                setTimeout(function () {
                    // arrived -> reveal the updated belief; pop in the new token
                    refreshBeliefs({ button: button, outcome: outcome });
                    setTimeout(function () {
                        // slide the agent back to the centre (animated) for the next press
                        const agentEl = document.getElementById("agent");
                        agent_topPos = topPos0;
                        agent_leftPos = leftPos0;
                        agentEl.style.top = topPos0 + "%";
                        agentEl.style.left = leftPos0 + "%";
                        setTimeout(function () {
                            if (ended) return;
                            trial_num += 1;
                            if (trial_num <= N_TRIALS) armPress();
                            else finishRoom();
                        }, MOVE_MS + 150);
                    }, 300);
                }, MOVE_MS);
            }

            function onCheck(rt) {
                // tick button: end sampling early and move on to the coin phase
                if (ended) return;
                sampling_ended = true; // retained for any external readers
                jsPsych.data.get().push(stampSession({
                    task: taskName,
                    room_num: room_num,
                    trial_num: trial_num,
                    ended_early: true,
                    rt: rt,
                    counts: countsSnapshot(),
                    posterior_means: posteriorSnapshot()
                }));
                finishRoom();
            }

            // arm the first press (controls are already fresh from the stimulus)
            wireButtons(onPress, onCheck);
        }
    };
}

//----------------------------------------------------------------------------//
// Gold pause: brief interlude showing a gold coin in the centre.
//----------------------------------------------------------------------------//
function make_gold_pause(room_num) {
    return {
        type: jsPsychHtmlKeyboardResponse,
        choices: "NO_KEYS",
        response_ends_trial: false,
        trial_duration: 1000,
        stimulus: screenHTML({
            title: "Gold time!",
            stage: `
                <div style="display:flex; justify-content:center; align-items:center; min-height:220px;">
                    ${goldCoinStaticHTML()}
                </div>`
        }),
        data: { task: "gold_pause", room_num: room_num }
    };
}

//----------------------------------------------------------------------------//
// Gold collection: a coin appears at a random reachable cell; the participant
// picks a button to try to reach it.
//   SHOW_GOLD_OUTCOME true  -> agent moves per the transition function, feedback shown.
//   SHOW_GOLD_OUTCOME false -> no outcome shown; move straight on to the next room.
//----------------------------------------------------------------------------//
function make_gold_trial(room_num, opts) {
    opts = opts || {};
    const buttonOrder = opts.buttonOrder || BUTTON_ORDER;
    return {
        type: jsPsychHtmlKeyboardResponse,
        choices: "NO_KEYS",
        stimulus: screenHTML({
            title: "Gold time!",
            lines: [
                `A gold coin appeared. Click the button most likely to reach it.`
            ],
            stage: `
                <div class="task-row">
                    ${checkButtonHTML({ placeholder: true })}
                    ${initialize_agent_gold()}
                    ${buttonStackHTML({ buttonOrder: buttonOrder })}
                    ${beliefPanelHTML()}
                </div>`
        }),
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
                // "correct" = chose the button with the objectively highest true
                // probability of reaching the coin (no sampling). Tally it for the
                // bonus. Ties (equal best true prob) count as correct for either.
                const chosen_gold_prob = TRUE_T[button][goldOutcome];
                const best_gold_prob = Math.max.apply(
                    null, BUTTONS.map(function (b) { return TRUE_T[b][goldOutcome]; })
                );
                const success = chosen_gold_prob === best_gold_prob;
                if (success) collected_gold += 1;

                const trial_data = {
                    chosen_button: button,
                    gold_outcome: goldOutcome,
                    rt: rt,
                    counts: countsSnapshot(),                 // final transition counts for the room
                    posterior_means: posteriorSnapshot(),     // belief the choice was based on
                    button_ctx: JSON.parse(JSON.stringify(BUTTON_CTX)), // prior each button came from
                    chosen_button_ctx: BUTTON_CTX[button],
                    chosen_gold_prob: chosen_gold_prob,       // true P(chosen button -> coin)
                    best_gold_prob: best_gold_prob,           // best true P(any button -> coin)
                    success: success,
                    collected_gold: collected_gold,
                    outcome_shown: SHOW_GOLD_OUTCOME
                };

                if (!SHOW_GOLD_OUTCOME) {
                    // do not reveal (agent stays put); on to the next room
                    var ppt_data_hidden = jsPsych.data.get().json();
                    send_incomplete(id, ppt_data_hidden);
                    setTimeout(() => jsPsych.finishTrial(trial_data), 1000);
                    return;
                }

                // reveal (only if SHOW_GOLD_OUTCOME): move the agent to the coin on a
                // correct choice, else to the chosen button's most likely location.
                const revealOutcome = success
                    ? goldOutcome
                    : OUTCOMES.reduce(function (bestO, o) {
                        return TRUE_T[button][o] > TRUE_T[button][bestO] ? o : bestO;
                    }, OUTCOMES[0]);
                moveAgent(revealOutcome);

                setTimeout(function () {
                    // once the agent has arrived, the result replaces the prompt above
                    // the room (as in the coin demos)
                    showScreenFeedback(success ? "You got the gold!" : "Missed it...", success);

                    // save data so far
                    var ppt_data = jsPsych.data.get().json();
                    send_incomplete(id, ppt_data);

                    setTimeout(() => jsPsych.finishTrial(trial_data), 1400);
                }, MOVE_MS);
            });
        }
    };
}
