//----------------------------------------------------------------------------//
// Shared building blocks
//----------------------------------------------------------------------------//
// The circle buttons: a triangle for 3 buttons, a diagonal pair for 2, with the
// colour->position mapping randomised so position isn't associated with a colour.
// buttonOrder parameter allows each room to capture its own layout independently:
// [top, bottomLeft, bottomRight] for the triangle, [upperLeft, lowerRight] for
// the diagonal. `ids: false` leaves out the element ids, for screens that show
// several rooms' buttons at once.
function buttonStackHTML(opts) {
    opts = opts || {};
    const label_on = opts.label_on === true;
    const withIds = opts.ids !== false;
    const order = opts.buttonOrder || BUTTON_ORDER;
    function labelFor(color) {
        return color.charAt(0).toUpperCase() + color.slice(1) + " button";
    }
    function item(color, posClass) {
        return `
            <div class="button-triangle-item ${posClass}">
                <div class="button-label${label_on ? "" : " hidden"}">${label_on ? labelFor(color) : ""}</div>
                <div class="cbtn ${color}"${withIds ? ` id="btn-${color}"` : ""}></div>
            </div>`;
    }
    if (order.length === 2) {
        return `
        <div class="button-diagonal">
            ${item(order[0], "pos-upper-left")}
            ${item(order[1], "pos-lower-right")}
        </div>`;
    }
    const [top, bl, br] = order;
    return `
        <div class="button-triangle">
            ${item(top, "pos-top")}
            ${item(bl, "pos-bottom-left")}
            ${item(br, "pos-bottom-right")}
        </div>`;
}

// The "done sampling" tick button, shown to the left of the room. With
// { placeholder: true } it renders the same element but invisible (keeping its
// layout footprint) so grids without a tick don't shift horizontally.
// With TERMINATE off there is no tick in this version of the task at all, so it
// renders nothing and the room keeps the space -- no screen has one, so nothing
// shifts relative to anything else.
function checkButtonHTML(opts) {
    if (!TERMINATE) return ``;
    const placeholder = opts && opts.placeholder;
    const tick_label = opts && opts.tick_label;
    return `
        <div class="check-stack${placeholder ? " hidden" : ""}"${placeholder ? " aria-hidden=\"true\"" : ""}>
            ${tick_label ? `<div class="check-label">Tick button</div>` : ""}
            <div class="checkbtn" id="btn-check"><img src="img/Check_orange.png" alt="Done testing"></div>
        </div>`;
}

// Right-hand belief panel: colorbar legend (overlay) or two belief grids (separate).
function beliefPanelHTML() {
    if (BELIEF_DISPLAY === "counters") return ``;
    if (BELIEF_DISPLAY === "overlay") {
        return `<div class="belief-stack">${BUTTONS.map(colorbarLegendHTML).join("")}</div>`;
    }
    return `<div class="belief-stack">${BUTTONS.map(beliefBlockHTML).join("")}</div>`;
}

// How many of the room's presses are still to come, e.g. "8 choices remaining".
function choicesRemainingText(remaining) {
    return `${remaining} choice${remaining === 1 ? "" : "s"} remaining`;
}

// The same, for the observation phase -- these are presses the participant watches
// rather than makes, so they are counted separately from their own choices.
function observationsRemainingText(remaining) {
    return `${remaining} choice${remaining === 1 ? "" : "s"} left to watch`;
}

// The progress bar at the top of the screen. It carries no numbers -- how far
// through the rooms you are is shown by the bar alone -- and is hidden outside the
// rooms themselves (instructions, questionnaires, exclusion screen). `fraction` is
// the proportion of rooms COMPLETED, so it reads 0 in the first room's intro and
// fills as rooms are finished.
function setRoomProgress(fraction) {
    const container = document.getElementById("jspsych-progressbar-container");
    if (container) container.classList.add("progress-visible");
    if (jsPsych.progressBar) {
        jsPsych.progressBar.progress = Math.max(0, Math.min(1, fraction));
    }
}

function hideRoomProgress() {
    const container = document.getElementById("jspsych-progressbar-container");
    if (container) container.classList.remove("progress-visible");
}

// Redraw the belief display (and the sample-counter titles) from current counts.
function refreshBeliefs(highlight, buttonOrder) {
    if (BELIEF_DISPLAY === "counters") {
        renderMainCounters(highlight, buttonOrder);
    } else if (BELIEF_DISPLAY === "overlay") {
        renderMainBeliefOverlay();
    } else {
        for (const b of BUTTONS) {
            renderBeliefGrid(b, document.getElementById("belief-" + b));
        }
    }
    updateBeliefLabels();
}

// Manually-pushed data rows don't get the session-level fields that jsPsych
// auto-applies to rows it writes itself, so stamp them on here to match. Shared by
// the observation phase and the participant's own presses.
function stampSession(row, buttonOrder) {
    buttonOrder = buttonOrder || BUTTON_ORDER;
    row.subject_id = id;  // use the internal id from the backend rather than the Prolific PID
    row.study_id = study_id;
    row.session_id = session_id;
    row.belief_display = BELIEF_DISPLAY;
    row.sample_cost = SAMPLE_COST;
    row.terminate = TERMINATE;
    row.objective = OBJECTIVE;
    row.confidence_on = CONFIDENCE;
    row.alpha = ALPHA;
    row.contextual = CONTEXTUAL;
    row.alpha_ctx1 = ALPHA_CTX1;
    row.alpha_ctx2 = ALPHA_CTX2;
    // which colour sits where: [top, bottomLeft, bottomRight] for the 3-button
    // triangle, [upperLeft, lowerRight] for the 2-button diagonal
    if (buttonOrder.length === 2) {
        row.button_upper_left = buttonOrder[0];
        row.button_lower_right = buttonOrder[1];
    } else {
        row.button_top = buttonOrder[0];
        row.button_bottom_left = buttonOrder[1];
        row.button_bottom_right = buttonOrder[2];
    }
    row.button_order = buttonOrder.slice(); // full array, for convenience downstream
    row.trial_type = "html-keyboard-response";
    return row;
}

// Spend one press worth of the room's gold coin and redraw it. No-op when
// SAMPLE_COST is 0 (the cost-free version of the design).
function chargeForPress() {
    if (SAMPLE_COST <= 0) return;
    GOLD_FRACTION = Math.max(0, GOLD_FRACTION - SAMPLE_COST);
    updateGoldCostCoin();
}

// Wire click handlers onto the buttons; `onPress(button, rt)` handles a colour
// choice. If a tick button is present and `onCheck` is given, `onCheck(rt)` fires
// when it is clicked. rt is measured from when the buttons become available.
function wireButtons(onPress, onCheck) {
    const btnEls = {};
    for (const b of BUTTONS) btnEls[b] = document.getElementById("btn-" + b);
    const btnCheck = document.getElementById("btn-check");
    const t0 = performance.now();

    function disableAll() {
        for (const b of BUTTONS) btnEls[b].classList.add("disabled");
        if (btnCheck) btnCheck.classList.add("disabled");
    }

    function handle(button) {
        const rt = Math.round(performance.now() - t0);
        disableAll();
        for (const b of BUTTONS) {
            if (b !== button) btnEls[b].classList.add("hidden");
        }
        if (btnCheck) (btnCheck.closest(".check-stack") || btnCheck).classList.add("hidden");
        onPress(button, rt);
    }
    for (const b of BUTTONS) {
        btnEls[b].addEventListener("click", () => handle(b));
    }

    if (btnCheck && onCheck) {
        btnCheck.addEventListener("click", function () {
            const rt = Math.round(performance.now() - t0);
            disableAll();
            onCheck(rt);
        });
    }
}

//----------------------------------------------------------------------------//
// The room's preset entry (canonical history + this room's colour/direction
// mapping + the shuffled press sequence). Built by loadPresets() in config.js.
//----------------------------------------------------------------------------//
function roomPreset(room_num) {
    const preset = ROOM_PRESETS[room_num - 1];
    if (!preset) throw new Error(`no preset for room ${room_num}`);
    return preset;
}

//----------------------------------------------------------------------------//
// Room intro: reset the tally and draw this room's hidden transition functions
// from the posterior implied by the history the participant is about to watch.
//----------------------------------------------------------------------------//
function make_room_intro(room_num) {
    const preset = roomPreset(room_num);
    return {
        type: jsPsychHtmlKeyboardResponse,
        stimulus: screenHTML({
            title: `New room`,
            // no "Room n of N": the progress bar at the top carries this instead
            lines: [
                `First, watch as ${preset.n_preset_presses} choices are made for you.`,
                `Then you will have ${N_REMAINING_TRIALS} choice${N_REMAINING_TRIALS === 1 ? "" : "s"} of your own.`,
                `Press any key to begin.`
            ]
        }),
        data: { task: "room_intro", room_num: room_num },
        on_start: function () {
            for (const b of BUTTONS) {
                for (const o of OUTCOMES) counts[b][o] = 0;
            }
            // rooms completed so far -- 0 at the start of room 1
            setRoomProgress((room_num - 1) / N_ROOMS);
            GOLD_FRACTION = 1;              // fresh coin for this room
            // hidden dynamics consistent with the observations about to be shown
            sampleTrueTFromPreset(preset.presetCounts);
            sampling_ended = false;
        },
        on_finish: function (data) {
            // log this room's (hidden) generative transition functions for analysis,
            // and which context each was drawn from (null unless CONTEXTUAL)
            data.true_T = JSON.parse(JSON.stringify(TRUE_T));
            data.button_ctx = JSON.parse(JSON.stringify(BUTTON_CTX));
            // and the preset that seeded it, with this room's concrete mapping
            data.preset_history = JSON.parse(JSON.stringify(preset.history));
            data.button_map = JSON.parse(JSON.stringify(preset.button_map));
            data.outcome_map = JSON.parse(JSON.stringify(preset.outcome_map));
            data.preset_counts = JSON.parse(JSON.stringify(preset.presetCounts));
            data.n_preset_presses = preset.n_preset_presses;
        }
    };
}

//----------------------------------------------------------------------------//
// Observation phase: the participant WATCHES the room's preset history play out.
// One press at a time -- the pressed button pulses, the agent moves to the
// observed outcome, a token lands there, the agent returns to the centre -- so
// the tally builds up exactly as it would if they had made the presses.
//
// Nothing is clickable (every button carries .inert). The whole phase is a single
// jsPsych trial, with one data row per observation pushed onto the collection
// (task "observe"/"practice_observe"), mirroring how make_room_sampling logs its
// presses.
//
// `opts.preset` supplies the history directly, for the instruction practices --
// they have no room of their own, so roomPreset() has nothing to look up.
//----------------------------------------------------------------------------//
function make_room_demo(room_num, opts) {
    opts = opts || {};
    const practice = opts.practice === true;
    const taskName = practice ? "practice_observe" : "observe";
    const buttonOrder = opts.buttonOrder || BUTTON_ORDER;
    const preset = opts.preset || roomPreset(room_num);
    const seq = preset.sequence;

    return {
        type: jsPsychHtmlKeyboardResponse,
        choices: "NO_KEYS",
        // a function, not a string: goldCostCoinHTML() reads GOLD_FRACTION, which is
        // only correct once the room has actually started
        stimulus: function () {
            return screenHTML({
                lines: [
                    `<span class="trial-counter">${observationsRemainingText(seq.length)}</span>`,
                    `Watch the choices being made for you.`
                ],
                gap: goldCostCoinHTML(),
                stage: `
                    <div class="task-row">
                        ${checkButtonHTML({ placeholder: true })}
                        ${initialize_agent()}
                        ${buttonStackHTML({ buttonOrder: buttonOrder })}
                        ${beliefPanelHTML()}
                    </div>`
            });
        },
        data: { task: "room_demo", room_num: room_num, practice: practice },
        on_start: function () {
            agent_topPos = topPos0;
            agent_leftPos = leftPos0;
        },
        on_load: function () {
            refreshBeliefs(null, buttonOrder);

            // nothing here is clickable: visually normal, inert to the mouse
            const btnEls = {};
            for (const b of BUTTONS) {
                btnEls[b] = document.getElementById("btn-" + b);
                btnEls[b].classList.add("inert");
            }
            const agentEl = document.getElementById("agent");
            const counter = document.querySelector(".trial-counter");

            let i = 0;

            function toCentre() {
                agent_topPos = topPos0;
                agent_leftPos = leftPos0;
                agentEl.style.top = topPos0 + "%";
                agentEl.style.left = leftPos0 + "%";
            }

            function step() {
                if (i >= seq.length) {
                    jsPsych.finishTrial({
                        task: "room_demo",
                        room_num: room_num,
                        practice: practice,
                        n_observed: seq.length
                    });
                    return;
                }

                const button = seq[i].button;
                const outcome = seq[i].outcome;
                const el = btnEls[button];
                if (counter) counter.textContent = observationsRemainingText(seq.length - i);

                el.classList.add("pressing");   // the choice being made for them
                setTimeout(function () {
                    el.classList.remove("pressing");

                    const counts_pre = countsSnapshot();
                    const posteriors_pre = posteriorSnapshot();
                    const gold_fraction_pre = GOLD_FRACTION;

                    // the observed presses cost the same as the participant's own,
                    // unless DEPLETE_DURING_DEMO says otherwise
                    if (DEPLETE_DURING_DEMO) chargeForPress();

                    moveAgent(outcome);

                    // token lands once the agent has arrived
                    setTimeout(function () {
                        counts[button][outcome] += 1;
                        refreshBeliefs({ button: button, outcome: outcome }, buttonOrder);

                        jsPsych.data.get().push(stampSession({
                            task: taskName,
                            room_num: room_num,
                            trial_num: i + 1,
                            observed_button: button,
                            observed_outcome: outcome,
                            rt: null,
                            counts: counts_pre,
                            posterior_means: posteriors_pre,
                            counts_post: countsSnapshot(),
                            gold_fraction_pre: gold_fraction_pre,
                            gold_fraction_post: GOLD_FRACTION
                        }, buttonOrder));
                    }, MOVE_MS);

                    setTimeout(function () {
                        toCentre();
                        i += 1;
                        setTimeout(step, DEMO_GAP_MS);
                    }, DEMO_VIEW_MS);
                }, DEMO_PRESS_MS);
            }

            setTimeout(step, 500);
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
// The tick ("done testing") ends the phase early -- it is the termination arm of
// the model, and stays available even when there is only one choice left. The
// trial ends when the participant has used all `nTrials` presses or clicks it.
// With TERMINATE off there is no tick, and the trial can only end by using up the
// press budget.
//
// `opts.nTrials` is the press budget: N_REMAINING_TRIALS in the preset design,
// N_TRIALS for the instruction demos that still play a whole room.
//
// `opts.tick: false` renders the tick as an invisible placeholder and leaves it
// unwired -- for the instruction practice that runs BEFORE the tick has been
// explained. It keeps its layout footprint, so the room does not shift when the
// tick appears on a later screen.
//----------------------------------------------------------------------------//
function make_room_sampling(room_num, opts) {
    opts = opts || {};
    const practice = opts.practice === true;
    const taskName = practice ? "practice_sample" : "sample";
    const buttonOrder = opts.buttonOrder || BUTTON_ORDER;
    const nTrials = opts.nTrials != null ? opts.nTrials : N_TRIALS;
    const withTick = TERMINATE && opts.tick !== false;
    // the coin carries over from the observation phase unless told to reset it
    const resetGold = opts.resetGold === true;

    return {
        type: jsPsychHtmlKeyboardResponse,
        choices: "NO_KEYS",
        // a function, not a string: the coin drawn here is whatever the observation
        // phase left of it, so it can only be built once the room is under way.
        // jsPsych evaluates parameters BEFORE on_start, so the resetGold case (the
        // practice room, which has no observation phase) is handled here too.
        stimulus: function () {
            if (resetGold) GOLD_FRACTION = 1;
            return screenHTML({
                lines: [
                    `<span class="trial-counter">${choicesRemainingText(nTrials)}</span>`,
                    withTick
                        ? `Now it is your turn. Click a button to move, or the tick to finish testing.`
                        : `Now it is your turn. Click a button to move.`
                ],
                gap: goldCostCoinHTML(),
                stage: `
                    <div class="task-row">
                        ${checkButtonHTML({ placeholder: !withTick })}
                        ${initialize_agent()}
                        ${buttonStackHTML({ buttonOrder: buttonOrder })}
                        ${beliefPanelHTML()}
                    </div>`
            });
        },
        data: { task: "room_sampling", room_num: room_num, practice: practice },
        on_start: function () {
            // agent starts in the central cell
            agent_topPos = topPos0;
            agent_leftPos = leftPos0;
            // NB: the coin is NOT reset here -- whatever the observation phase spent
            // stays spent (make_room_intro is what starts each room at a full coin)
        },
        on_load: function () {
            refreshBeliefs(null, buttonOrder);

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
                const bs = row.querySelector(".button-triangle");
                if (bs) bs.outerHTML = buttonStackHTML({ buttonOrder: buttonOrder });
                const cs = row.querySelector(".check-stack");
                if (cs) cs.outerHTML = checkButtonHTML({ placeholder: !withTick });
                const counter = document.querySelector(".trial-counter");
                if (counter) counter.textContent = choicesRemainingText(nTrials - trial_num + 1);
                wireButtons(onPress, withTick ? onCheck : null);
            }

            function onPress(button, rt) {
                const counts_pre = countsSnapshot();
                const posteriors_pre = posteriorSnapshot();
                const gold_fraction_pre = GOLD_FRACTION;

                // sampling cost: each click removes a SAMPLE_COST slice of the coin
                // (no-op when SAMPLE_COST is 0)
                chargeForPress();

                const outcome = sampleCategorical(TRUE_T[button]);
                counts[button][outcome] += 1;

                moveAgent(outcome);

                jsPsych.data.get().push(stampSession({
                    task: taskName,
                    room_num: room_num,
                    trial_num: trial_num,
                    chosen_button: button,
                    outcome: outcome,
                    rt: rt,
                    ended_early: false,
                    counts: counts_pre,
                    posterior_means: posteriors_pre,
                    counts_post: countsSnapshot(),
                    gold_fraction_pre: gold_fraction_pre,      // <-- new
                    gold_fraction_post: GOLD_FRACTION          // <-- new
                }));

                setTimeout(function () {
                    refreshBeliefs({ button: button, outcome: outcome }, buttonOrder);
                    setTimeout(function () {
                        const agentEl = document.getElementById("agent");
                        agent_topPos = topPos0;
                        agent_leftPos = leftPos0;
                        agentEl.style.top = topPos0 + "%";
                        agentEl.style.left = leftPos0 + "%";
                        setTimeout(function () {
                            if (ended) return;
                            trial_num += 1;
                            if (trial_num <= nTrials) armPress();
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
            wireButtons(onPress, withTick ? onCheck : null);
        }
    };
}

//----------------------------------------------------------------------------//
// Room panel: one room with its buttons BELOW it. `roomHTML` is the room itself:
// initialize_agent_gold() on the gold screen, choiceRoomHTML() on the choice
// screen. Panels sit in a .choice-row, which sets the (smaller) room size.
// `opts.idle` greys the panel out, as for a room not currently being played.
//----------------------------------------------------------------------------//
function roomPanelHTML(roomHTML, buttonOrder, opts) {
    opts = opts || {};
    return `
        <div class="room-panel${opts.idle ? " panel-idle" : ""}">
            ${roomHTML}
            ${buttonStackHTML({ buttonOrder: buttonOrder, ids: opts.ids })}
        </div>`;
}

// The room as drawn on the choice screen: as initialize_agent(), but without ids,
// since N_AFC copies of it share the screen. Its parts are found by class within
// their .room-panel instead.
function choiceRoomHTML() {
    return `
    <div class="container">
        <img src="img/BaseAction_4k.png" alt="Base Action Image" class="base-image" decoding="sync" fetchpriority="high">
        <div class="belief-overlay"></div>
        <img src="img/Agent.png" alt="Agent Image" class="agent-image" decoding="sync">
    </div>`;
}

//----------------------------------------------------------------------------//
// Choice intro: reset the coin and say what is coming. The hidden transition
// functions are only drawn once a room has been chosen (make_room_choice).
//----------------------------------------------------------------------------//
function make_choice_intro(choice_num) {
    return {
        type: jsPsychHtmlKeyboardResponse,
        stimulus: screenHTML({
            title: `New choice`,
            // no "Choice n of N": the progress bar at the top carries this instead
            lines: [
                `First, watch as some choices are made for you in each of the ${N_AFC} rooms.`,
                OBJECTIVE === "gold"
                    ? `Then click on the room you want to enter to collect the gold.`
                    : `Then click on the room in which you feel the greatest sense of control.`,
                `Press any key to begin.`
            ].map(line => `<span class="screen-question">${line}</span>`)
        }),
        data: { task: "choice_intro", choice_num: choice_num },
        on_start: function () {
            // choices completed so far -- 0 at the start of the first
            setRoomProgress((choice_num - 1) / N_ROOMS);
            GOLD_FRACTION = 1;              // fresh coin for this choice
            CHOSEN = null;
        }
    };
}

//----------------------------------------------------------------------------//
// Room choice: the N_AFC rooms of this choice are shown side by side, empty.
// Then each room's preset history plays out in turn, one press at a time, as in
// make_room_demo -- the room being played is shown normally and the others are
// dimmed. Once every room has played, the participant clicks the room they want
// to enter. That room's tally becomes `counts`, its hidden transition functions
// are drawn from the posterior given its history, and the gold phase follows in it.
//
// The whole screen is a single jsPsych trial (no DOM teardown, so no flicker),
// with one "observe" data row pushed per press, tagged with the room it was in.
//
// For the instructions, which have no trial of their own in the presets file:
// `opts.presets` / `opts.orders` / `opts.optionOrder` supply the rooms directly;
// `opts.practice` renames the data rows and skips saving progress; and
// `opts.watchOnly` ends on a Next button once the rooms have played, instead of
// offering the choice.
//----------------------------------------------------------------------------//
function make_room_choice(choice_num, opts) {
    opts = opts || {};
    const practice = opts.practice === true;
    const presets = opts.presets || ROOM_PRESETS[choice_num - 1];              // by option
    const orders = opts.orders || ROOM_BUTTON_ORDERS[choice_num - 1];          // by option
    const optionOrder = opts.optionOrder || OPTION_ORDERS[choice_num - 1];     // position -> option
    const taskName = practice ? "practice_room_choice" : "room_choice";

    // the prompt while the room at screen position `pos` is played: left/right, or
    // left/middle/right
    function watchLine(pos) {
        const n = optionOrder.length;
        const side = n === 2 ? ["left", "right"][pos]
            : n === 3 ? ["left", "middle", "right"][pos]
            : `${pos + 1}${["st", "nd", "rd"][pos] || "th"}`;
        return `<span class="screen-question">Watch the choices being made for you in the <strong>${side}</strong> room.</span>`;
    }

    return {
        type: jsPsychHtmlKeyboardResponse,
        choices: "NO_KEYS",
        stimulus: screenHTML({
            // opens on the first room's prompt, so the text does not change as it starts
            lines: [watchLine(0)],
            stage: `
                <div class="choice-row">
                    ${optionOrder.map(j => roomPanelHTML(choiceRoomHTML(), orders[j], { ids: false })).join("")}
                </div>`
        }) + (opts.watchOnly ? navHTML("Next") : ``),
        data: { task: taskName, choice_num: choice_num, practice: practice },
        on_load: function () {
            // panels are in screen order; roomCounts is by option
            const panels = Array.from(document.querySelectorAll(".choice-row .room-panel"));
            const roomCounts = presets.map(() => freshButtonMap());
            panels.forEach(function (panel, pos) {
                const j = optionOrder[pos];
                // nothing here is clickable: visually normal, inert to the mouse
                panel.querySelectorAll(".cbtn").forEach(el => el.classList.add("inert"));
                renderCountersInto(panel.querySelector(".belief-overlay"), roomCounts[j], null, orders[j]);
            });

            // play the room at screen position `pos`, then the next one
            function playRoom(pos) {
                if (pos >= panels.length) {
                    if (opts.watchOnly) offerNext();
                    else offerChoice();
                    return;
                }
                const j = optionOrder[pos];
                const panel = panels[pos];
                const seq = presets[j].sequence;
                const cnts = roomCounts[j];
                const layer = panel.querySelector(".belief-overlay");
                const agentEl = panel.querySelector(".agent-image");
                panels.forEach((p, q) => p.classList.toggle("panel-idle", q !== pos));
                const lines = document.getElementById("screen-lines");
                if (lines) lines.innerHTML = linesHTML([watchLine(pos)]);
                // countsSnapshot/posteriorSnapshot read the global tally
                counts = cnts;

                let i = 0;

                function toCentre() {
                    agentEl.style.top = topPos0 + "%";
                    agentEl.style.left = leftPos0 + "%";
                }

                function step() {
                    if (i >= seq.length) {
                        setTimeout(() => playRoom(pos + 1), CHOICE_ROOM_GAP_MS);
                        return;
                    }
                    const button = seq[i].button;
                    const outcome = seq[i].outcome;
                    const el = panel.querySelector(".cbtn." + button);

                    el.classList.add("pressing");   // the choice being made for them
                    setTimeout(function () {
                        el.classList.remove("pressing");
                        const counts_pre = countsSnapshot();
                        const posteriors_pre = posteriorSnapshot();
                        const obs_num = i + 1;

                        moveAgent(outcome, agentEl);

                        // token lands once the agent has arrived
                        setTimeout(function () {
                            cnts[button][outcome] += 1;
                            renderCountersInto(layer, cnts, { button: button, outcome: outcome }, orders[j]);

                            jsPsych.data.get().push(stampSession({
                                task: practice ? "practice_observe" : "observe",
                                choice_num: choice_num,
                                option: j,
                                position: pos,
                                obs_num: obs_num,
                                observed_button: button,
                                observed_outcome: outcome,
                                rt: null,
                                counts: counts_pre,
                                posterior_means: posteriors_pre,
                                counts_post: countsSnapshot()
                            }, orders[j]));
                        }, MOVE_MS);

                        setTimeout(function () {
                            toCentre();
                            i += 1;
                            setTimeout(step, DEMO_GAP_MS);
                        }, DEMO_VIEW_MS);
                    }, DEMO_PRESS_MS);
                }

                setTimeout(step, 500);
            }

            // watch-only (instructions): every room has played, so just move on
            function offerNext() {
                panels.forEach(p => p.classList.remove("panel-idle"));
                const cont = document.getElementById("demo-continue");
                cont.style.display = "inline-block";
                cont.addEventListener("click", function () {
                    jsPsych.finishTrial({ task: taskName, choice_num: choice_num, practice: practice });
                });
            }

            // every room has played: let the participant pick one
            function offerChoice() {
                const lines = document.getElementById("screen-lines");
                if (lines) lines.innerHTML = linesHTML([
                    OBJECTIVE === "gold"
                        ? `<span class="screen-question">Click on the room you want to enter.</span>`
                        : `<span class="screen-question">In which room do you feel the greatest <strong>sense of control</strong>?</span>`
                ]);
                const t0 = performance.now();
                let chosen = false;

                panels.forEach(function (panel, pos) {
                    panel.classList.remove("panel-idle");
                    panel.classList.add("panel-selectable");
                    panel.addEventListener("click", function () {
                        if (chosen) return;
                        chosen = true;
                        const rt = Math.round(performance.now() - t0);
                        const j = optionOrder[pos];

                        // the gold phase runs in the chosen room
                        counts = roomCounts[j];
                        sampleTrueTFromPreset(presets[j].presetCounts);
                        CHOSEN = { option: j, position: pos, preset: presets[j], buttonOrder: orders[j] };

                        panels.forEach(function (p, q) {
                            p.classList.remove("panel-selectable");
                            if (q !== pos) p.classList.add("panel-idle");
                        });

                        const copy = x => JSON.parse(JSON.stringify(x));
                        setTimeout(function () {
                            // under "control" with no confidence rating this is the
                            // trial's last screen, so save progress here
                            if (OBJECTIVE === "control" && !CONFIDENCE && !practice) {
                                send_incomplete(id, jsPsych.data.get().json());
                            }
                            jsPsych.finishTrial({
                                rt: rt,
                                chosen_option: j,
                                chosen_position: pos,
                                // which trial of the presets file this was
                                preset_trial_index: practice ? null : TRIAL_ORDER[choice_num - 1],
                                option_order: optionOrder.slice(),
                                // per option, indexed as in the presets file
                                preset_histories: copy(presets.map(p => p.history)),
                                button_maps: copy(presets.map(p => p.button_map)),
                                outcome_maps: copy(presets.map(p => p.outcome_map)),
                                preset_counts: copy(presets.map(p => p.presetCounts)),
                                n_preset_presses: presets.map(p => p.n_preset_presses),
                                button_orders: copy(orders),
                                // the chosen room's tally and (hidden) generative T
                                counts: countsSnapshot(),
                                true_T: copy(TRUE_T),
                                button_ctx: copy(BUTTON_CTX)
                            });
                        }, 500);
                    });
                });
            }

            setTimeout(() => playRoom(0), CHOICE_PREVIEW_MS);
        }
    };
}

//----------------------------------------------------------------------------//
// Confidence (only under CONFIDENCE): the chosen room is shown on its own, with
// its tally and buttons, and the participant rates on a 0-100 slider how likely
// they think they are to get the coin in it. The slider starts in the middle and
// must be moved before Continue is enabled, so a rating of 50 is always a
// deliberate one.
//
// For the instructions: `opts.title` / `opts.lines` add explanatory text above the
// question, and `opts.question: false` leaves the question out; `opts.preset` (+ `opts.buttonOrder`) shows that preset's full tally
// rather than the chosen room's; `opts.practice` renames the data row and skips
// saving progress.
//----------------------------------------------------------------------------//
function make_confidence_trial(choice_num, opts) {
    opts = opts || {};
    const practice = opts.practice === true;
    const taskName = practice ? "practice_confidence" : "confidence";
    let buttonOrder = BUTTON_ORDER;
    return {
        type: jsPsychHtmlKeyboardResponse,
        choices: "NO_KEYS",
        // a function: the chosen room is only known once make_room_choice has run
        stimulus: function () {
            buttonOrder = opts.buttonOrder || (CHOSEN && CHOSEN.buttonOrder) || BUTTON_ORDER;
            return screenHTML({
                title: opts.title,
                lines: (opts.lines || []).concat(opts.question === false ? [] : [
                    OBJECTIVE === "gold"
                        ? `<span class="screen-question">How likely do you think you are to obtain a coin in this room?</span>`
                        : `<span class="screen-question">How strongly do you feel a <strong>sense of control</strong> in this room?</span>`
                ]),
                // the room on its own (no buttons -- nothing is pressed here), with
                // the slider underneath it
                stage: `
                    <div class="choice-row">
                        <div class="room-panel">
                            ${initialize_agent()}
                            <div class="confidence-wrap">
                                <input type="range" id="confidence-slider" class="confidence-slider" min="0" max="100" step="1" value="50">
                                <div class="confidence-labels">${OBJECTIVE === "gold"
                                    ? `<span>Very unlikely</span><span>Very likely</span>`
                                    : `<span>Not at all</span><span>Very strongly</span>`}</div>
                            </div>
                        </div>
                    </div>
                    <div class="screen-nav">
                        <button id="confidence-continue" class="jspsych-btn" disabled>Continue</button>
                    </div>`
            });
        },
        data: { task: taskName, choice_num: choice_num, practice: practice },
        on_start: function () {
            agent_topPos = topPos0;
            agent_leftPos = leftPos0;
            if (opts.preset) counts = JSON.parse(JSON.stringify(opts.preset.presetCounts));
        },
        on_load: function () {
            refreshBeliefs(null, buttonOrder);

            const slider = document.getElementById("confidence-slider");
            const cont = document.getElementById("confidence-continue");
            const t0 = performance.now();
            let moved = false;
            slider.addEventListener("input", function () {
                moved = true;
                cont.disabled = false;
            });
            cont.addEventListener("click", function () {
                if (!moved) return;
                // under "control" there is no gold trial to save progress, so the
                // trial's last screen does it instead
                if (OBJECTIVE === "control" && !practice) send_incomplete(id, jsPsych.data.get().json());
                jsPsych.finishTrial({
                    confidence: Number(slider.value),     // 0-100
                    rt: Math.round(performance.now() - t0),
                    chosen_option: (CHOSEN && !opts.preset) ? CHOSEN.option : null,
                    chosen_position: (CHOSEN && !opts.preset) ? CHOSEN.position : null,
                    counts: countsSnapshot()
                });
            });
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
        stimulus: function () {
            return screenHTML({
                title: "Gold time!",
                stage: `
                    <div style="display:flex; justify-content:center; align-items:center; min-height:220px;">
                        ${goldCoinStaticHTML(GOLD_FRACTION)}
                    </div>`
            });
        },
        data: { task: "gold_pause", room_num: room_num }
    };
}

//----------------------------------------------------------------------------//
// Gold collection: a coin appears at a random reachable cell; the participant
// picks a button to try to reach it.
//   SHOW_GOLD_OUTCOME true  -> agent moves per the transition function, feedback shown.
//   SHOW_GOLD_OUTCOME false -> no outcome shown; move straight on to the next room.
//----------------------------------------------------------------------------//
// Draws a single outcome from a button's true transition distribution.
// transitionProbs: object mapping outcome -> probability (should sum to ~1 over OUTCOMES)
function sampleOutcome(transitionProbs) {
    const r = Math.random();
    let cumulative = 0;
    for (const o of OUTCOMES) {
        cumulative += transitionProbs[o];
        if (r < cumulative) return o;
    }
    return OUTCOMES[OUTCOMES.length - 1]; // floating-point fallback
}

function make_gold_trial(room_num, opts) {
    opts = opts || {};
    const practice = opts.practice === true;
    const taskName = practice ? "practice_gold" : "gold";
    // the chosen room's layout is only known once make_room_choice has run, so it
    // is resolved when the stimulus is built (jsPsych evaluates it before on_load)
    let buttonOrder = opts.buttonOrder || BUTTON_ORDER;
    return {
        type: jsPsychHtmlKeyboardResponse,
        choices: "NO_KEYS",
        stimulus: function () {
            buttonOrder = (CHOSEN && CHOSEN.buttonOrder) || opts.buttonOrder || BUTTON_ORDER;
            return screenHTML({
                title: "Gold time!",
                lines: [
                    `A gold coin appeared. Click the button most likely to reach it.`
                ],
                // the practice reveals the outcome, so it carries the same aside as
                // the gold demos
                below: practice ? `<p class="screen-note">(Note that in the real experiment, you will not see whether or not you actually reached the gold.)</p>` : ``,
                stage: `
                    <div class="choice-row">
                        ${roomPanelHTML(initialize_agent_gold(), buttonOrder)}
                    </div>`
            });
        },
        data: { task: "gold", room_num: room_num, choice_num: room_num },
        on_start: function () {
            agent_topPos = topPos0;
            agent_leftPos = leftPos0;
        },
        on_load: function () {
            // gold appears at a random reachable (cardinal) cell
            const goldOutcome = OUTCOMES[Math.floor(Math.random() * OUTCOMES.length)];
            placeGold(goldOutcome);
            refreshBeliefs(null, buttonOrder);

            wireButtons(function (button, rt) {
                // "correct" = chose the button with the objectively highest true
                // probability of reaching the coin (no sampling). Purely a measure
                // of decision quality against the true model. Ties count as correct.
                const chosen_gold_prob = TRUE_T[button][goldOutcome];
                const best_gold_prob = Math.max.apply(
                    null, BUTTONS.map(function (b) { return TRUE_T[b][goldOutcome]; })
                );
                const correct = chosen_gold_prob === best_gold_prob;

                // "collected_gold" = an actual draw from the chosen button's true
                // transition distribution. This is what really happens in the room,
                // independent of whether the choice was the optimal one.
                const sampled_outcome = sampleOutcome(TRUE_T[button]);
                const collected = sampled_outcome === goldOutcome;
                if (collected && !practice) collected_gold += GOLD_FRACTION;

                const trial_data = {
                    chosen_button: button,
                    gold_outcome: goldOutcome,
                    rt: rt,
                    task: taskName,
                    practice: practice,
                    counts: countsSnapshot(),
                    posterior_means: posteriorSnapshot(),
                    button_ctx: JSON.parse(JSON.stringify(BUTTON_CTX)),
                    chosen_button_ctx: BUTTON_CTX[button],
                    chosen_gold_prob: chosen_gold_prob,
                    best_gold_prob: best_gold_prob,
                    correct: correct,
                    sampled_outcome: sampled_outcome,
                    collected: collected,
                    collected_gold_total: collected_gold,
                    outcome_shown: SHOW_GOLD_OUTCOME,
                    // which of the choice's rooms this is (null outside the choice task)
                    chosen_option: CHOSEN ? CHOSEN.option : null,
                    chosen_position: CHOSEN ? CHOSEN.position : null,
                    button_order: buttonOrder.slice(),
                    gold_fraction_remaining: GOLD_FRACTION   // <-- new: how much coin was left to win
                };

                if (!SHOW_GOLD_OUTCOME && !practice) {
                    // do not reveal (agent stays put); on to the next room
                    var ppt_data_hidden = jsPsych.data.get().json();
                    send_incomplete(id, ppt_data_hidden);
                    setTimeout(() => jsPsych.finishTrial(trial_data), 1000);
                    return;
                } else {

                    // reveal (only if SHOW_GOLD_OUTCOME): move the agent to wherever the
                    // sampled transition actually sent them — this is the true stochastic
                    // consequence of the button press, regardless of whether it was "correct".
                    moveAgent(sampled_outcome);

                    setTimeout(function () {
                        // once the agent has arrived, the result replaces the prompt above
                        // the room (as in the coin demos)
                        if (practice) {
                            const extraLines = [
                                `Press any key to continue.`
                            ];
                            showScreenFeedback(collected ? "You got the gold!" : "Missed it...", collected, extraLines);
                        } else {
                            const extraLines = []
                            showScreenFeedback(collected ? "You got the gold!" : "Missed it...", collected, extraLines);
                        }

                        // save data so far
                        var ppt_data = jsPsych.data.get().json();
                        send_incomplete(id, ppt_data);


                        // if practice, show the feedback until the participant presses a key; otherwise, move on after a short pause
                        if (practice) {
                            document.addEventListener("keydown", function onKey() {
                                document.removeEventListener("keydown", onKey);
                                jsPsych.finishTrial(trial_data);
                            });
                        } else {
                            setTimeout(() => jsPsych.finishTrial(trial_data), 1400);
                        }
                        
                    }, MOVE_MS);
                }
            });
        }
    };
}
