//----------------------------------------------------------------------------//
// Instructions phase: slide-based text (jsPsychInstructions, next/prev) with
// interactive demo trials interleaved so participants practise clicking a button
// and watch a coloured token appear on the location the agent reached.
//----------------------------------------------------------------------------//

//----------------------------------------------------------------------------//
// Static-HTML helpers (pure strings, no runtime DOM wiring) for illustrations
// shown inside instruction slides.
//----------------------------------------------------------------------------//
function cellPosStyle(idx, inset, size) {
    const cellPct = 100 / gridSize;
    const r = Math.floor(idx / gridSize);
    const c = idx % gridSize;
    return `left:${c * cellPct + cellPct * inset}%; top:${r * cellPct + cellPct * inset}%;` +
           `width:${cellPct * size}%; height:${cellPct * size}%;`;
}

// posterior mean from an arbitrary per-outcome count object {up,right,down,left}
function postFromCounts(cntObj, outcome) {
    const total = OUTCOMES.reduce((acc, o) => acc + cntObj[o], 0);
    return (ALPHA + cntObj[outcome]) / (K_OUTCOMES * ALPHA + total);
}

// the main room with the K reachable cells highlighted (no heatmaps)
function reachableRoomStaticHTML(goldOutcome) {
    let cells = "";
    // for (const outcome of OUTCOMES) {
    //     cells += `<div class="reachable-cell" style="${cellPosStyle(OUTCOME_CELL[outcome], 0.08, 0.84)}"></div>`;
    // }
    return `
        <div class="container">
            <img src="img/BaseAction_4k.png" alt="Base" class="base-image">
            <div class="belief-overlay">${cells}</div>
            ${goldOutcome ? goldInRoomStaticHTML(goldOutcome) : ""}
            <img src="img/Agent.png" alt="Agent" class="agent-image">
        </div>`;
}

// static belief grid for a given count object
function beliefGridStaticHTML(button, cntObj) {
    const cellOutcome = {};
    for (const outcome of OUTCOMES) cellOutcome[OUTCOME_CELL[outcome]] = outcome;

    let cells = "";
    for (let j = 0; j < gridSize * gridSize; j++) {
        if (j === 4) {
            cells += `<div class="belief-cell"><div class="belief-dot" style="background:rgb(${BTN_COLOR[button]})"></div></div>`;
        } else if (cellOutcome[j]) {
            const p = postFromCounts(cntObj, cellOutcome[j]);
            cells += `<div class="belief-cell" style="background:rgba(${BTN_COLOR[button]}, ${p})"><span class="belief-num">${p.toFixed(2)}</span></div>`;
        } else {
            cells += `<div class="belief-cell"></div>`;
        }
    }
    return `<div class="belief-grid">${cells}</div>`;
}

// static belief block (label + grid + colorbar)
function beliefBlockStaticHTML(button, label, cntObj) {
    return `
        <div class="belief-block">
            <div class="belief-label">${label}</div>
            <div class="belief-body">
                ${beliefGridStaticHTML(button, cntObj)}
                ${colorbarHTML(button)}
            </div>
        </div>`;
}

// The task display exactly as a participant meets it in a real room: the same
// tick, room, buttons and belief display, in the same places -- but inert, so
// nothing can be clicked and nothing animates. Counts default to an untouched
// room. Adapts to BELIEF_DISPLAY (counters -> tokens in the main grid; else two
// heatmaps beside it).
//
// opts.tick shows the real tick button. Slides that have not introduced it yet
// get an invisible placeholder instead: it keeps its layout footprint, so the
// room sits exactly where the task puts it, without offering the participant a
// control nothing has explained.
function taskDisplayStaticHTML(countsByButton, opts) {
    opts = opts || {};
    const zero = { up: 0, right: 0, down: 0, left: 0 };
    const cntByButton = {};
    for (const b of BUTTONS) {
        cntByButton[b] = (countsByButton && countsByButton[b]) || zero;
    }
    const tick = checkButtonHTML({ placeholder: opts.tick !== true, tick_label: opts.tick_label });
    const buttonStack = buttonStackHTML({ label_on: opts.label_on === true, buttonOrder: opts.buttonOrder });
    const goldOutcome = opts.goldOutcome;

    if (BELIEF_DISPLAY === "counters") {
        return `
            <div class="task-row" style="pointer-events:none;">
                ${tick}
                ${roomCountersStaticHTML(cntByButton, opts.buttonOrder, goldOutcome)}
                ${buttonStack}
            </div>`;
    }
    return `
        <div class="task-row" style="pointer-events:none;">
            ${tick}
            ${reachableRoomStaticHTML(goldOutcome)}
            ${buttonStack}
            <div class="belief-stack">
                ${BUTTONS.map(b =>
                    beliefBlockStaticHTML(b, b.charAt(0).toUpperCase() + b.slice(1) + " button", cntByButton[b])
                ).join("")}
            </div>
        </div>`;
}

// fraction defaults to 1 (full coin) so existing callers are unaffected.
function goldCoinStaticHTML(fraction) {
    const f = fraction === undefined ? 1 : fraction;
    return `
        <img src="img/Goal.png" alt="Gold Coin" class="gold-coin-static"
             style="${goldMaskStyle(f)}">`;
}

// The demo trials' own advance button. It is pinned to the bottom of the screen
// (see .screen-nav), in the same place as the instructions plugin's nav, so the
// room sits identically whether a slide or a demo is showing.
function navHTML(label) {
    return `
        <div class="screen-nav">
            <button id="demo-continue" class="demo-continue" style="display:none;">${label}</button>
        </div>`;
}

// the outcome a button's posterior currently favours (used by the coin demo)
function argmaxOutcome(button) {
    let best = OUTCOMES[0];
    let bestP = -1;
    for (const outcome of OUTCOMES) {
        const p = posteriorMean(button, outcome);
        if (p > bestP) { bestP = p; best = outcome; }
    }
    return best;
}

// the cardinal location with the most ("max") or fewest ("min") total tokens
// across both buttons -- used to place the practice coins in a well- vs poorly-
// sampled cell. Ties resolve to the first outcome in OUTCOMES order.
function outcomeByTokenCount(which) {
    let best = OUTCOMES[0];
    let bestVal = null;
    for (const outcome of OUTCOMES) {
        const total = BUTTONS.reduce((acc, b) => acc + counts[b][outcome], 0);
        if (bestVal === null ||
            (which === "max" && total > bestVal) ||
            (which === "min" && total < bestVal)) {
            bestVal = total;
            best = outcome;
        }
    }
    return best;
}

//----------------------------------------------------------------------------//
// Auto-play demo: the computer presses one button `cfg.outcomes.length` times,
// showing the agent move to each (fixed) outcome and return to the centre. Used
// to illustrate how reliable vs variable a button can be. A coloured token is
// placed on each location the agent reaches, so the tally builds up as the demo
// plays (the tally starts empty). The Next button appears after `cfg.revealAfter`
// observations (default 5) so the participant may continue early while the rest
// keep playing.
//----------------------------------------------------------------------------//
function make_auto_demo_trial(cfg) {
    const revealAfter = cfg.revealAfter || 5;
    return {
        type: jsPsychHtmlKeyboardResponse,
        choices: "NO_KEYS",
        stimulus: screenHTML({
            title: cfg.title,
            lines: cfg.lines,
            stage: `
                <div class="task-row">
                    ${checkButtonHTML({ placeholder: true })}
                    ${initialize_agent()}
                    ${buttonStackHTML()}
                    ${beliefPanelHTML()}
                </div>`
        }) + navHTML("Next"),
        data: { task: "auto_demo", demo_button: cfg.button },
        on_start: function () {
            agent_topPos = topPos0;
            agent_leftPos = leftPos0;
            // start from an empty tally so the tokens accumulate during the demo,
            // unless cfg.reset === false (the 2nd demo keeps the 1st demo's tokens)
            if (cfg.reset !== false) {
                for (const b of BUTTONS) {
                    for (const o of OUTCOMES) counts[b][o] = 0;
                }
            }
        },
        on_load: function () {
            refreshBeliefs();

            const active = document.getElementById("btn-" + cfg.button);
            const others = BUTTONS.filter(b => b !== cfg.button)
                .map(b => document.getElementById("btn-" + b));
            active.classList.add("inert");
            others.forEach(el => {
                el.classList.add("inert");
                el.style.opacity = "0.2";
            });

            const cont = document.getElementById("demo-continue");
            const agentEl = document.getElementById("agent");
            const seq = cfg.outcomes;
            let i = 0;
            let stopped = false; // set when the participant clicks Next early

            function toCentre() {
                agent_topPos = topPos0;
                agent_leftPos = leftPos0;
                agentEl.style.top = topPos0 + "%";
                agentEl.style.left = leftPos0 + "%";
            }
            function step() {
                if (stopped || i >= seq.length) return;
                active.classList.add("pressing"); // the computer "presses" the button
                setTimeout(function () {
                    if (stopped) return;
                    active.classList.remove("pressing");
                    const outcome = seq[i];
                    moveAgent(outcome);
                    i += 1;
                    // let the participant continue once they've seen enough
                    if (i >= revealAfter) cont.style.display = "inline-block";
                    // place the token once the agent has reached the location
                    setTimeout(function () {
                        if (stopped) return;
                        counts[cfg.button][outcome] += 1;
                        refreshBeliefs({ button: cfg.button, outcome: outcome });
                    }, MOVE_MS);
                    setTimeout(function () {
                        if (stopped) return;
                        toCentre();
                        setTimeout(step, 550); // pause at the centre before the next press
                    }, 800); // view the outcome
                }, 450); // press pulse before the move
            }
            setTimeout(step, 500);

            cont.addEventListener("click", function () {
                stopped = true; // halt any in-flight animation before the trial ends
                jsPsych.finishTrial({ task: "auto_demo", demo_button: cfg.button, observations: i });
            });
        }
    };
}

//----------------------------------------------------------------------------//
// Intro click demo: the participant makes the first press themselves, the room
// animates, and the resulting token remains visible for the following slide.
//----------------------------------------------------------------------------//
function make_intro_click_demo_trial() {
    return {
        type: jsPsychHtmlKeyboardResponse,
        choices: "NO_KEYS",
        stimulus: screenHTML({
            title: `How do the buttons work?`,
            lines: [
                OBJECTIVE === "gold"
                    ? `Fortunately, before the gold appears, you can test the buttons to learn how they work.`
                    : `To learn how the buttons work, you can test them.`,
                `NOTE: neither the colour nor the position of the buttons has <strong>any relation</strong> to the locations they reach.`,
                `To begin, <strong>click one of the buttons</strong> to see what happens.`,
            ],
            stage: `
                <div class="task-row">
                    ${checkButtonHTML({ placeholder: true })}
                    ${initialize_agent()}
                    ${buttonStackHTML()}
                    ${beliefPanelHTML()}
                </div>`
        }),
        data: { task: "instructions_intro_demo" },
        on_start: function () {
            for (const b of BUTTONS) {
                for (const o of OUTCOMES) counts[b][o] = 0;
            }
            sampleTrueT();
            sampling_ended = false;
            agent_topPos = topPos0;
            agent_leftPos = leftPos0;
        },
        on_load: function () {
            refreshBeliefs();

            const btnEls = {};
            for (const b of BUTTONS) btnEls[b] = document.getElementById("btn-" + b);
            const agentEl = document.getElementById("agent");

            function toCentre() {
                agent_topPos = topPos0;
                agent_leftPos = leftPos0;
                agentEl.style.top = topPos0 + "%";
                agentEl.style.left = leftPos0 + "%";
            }

            function handle(button) {
                for (const b of BUTTONS) btnEls[b].classList.add("disabled");
                for (const b of BUTTONS) {
                    if (b !== button) btnEls[b].classList.add("hidden");
                }

                const outcome = sampleCategorical(TRUE_T[button]);
                counts[button][outcome] += 1;
                moveAgent(outcome);

                setTimeout(function () {
                    refreshBeliefs({ button: button, outcome: outcome });
                }, MOVE_MS);

                setTimeout(function () {
                    toCentre();
                }, 800);

                setTimeout(function () {
                    jsPsych.finishTrial({
                        task: "instructions_intro_demo",
                        chosen_button: button,
                        outcome: outcome
                    });
                }, 1350);
            }

            for (const b of BUTTONS) {
                btnEls[b].addEventListener("click", () => handle(b));
            }
        }
    };
}

function make_testing_cost_demo_trial() {
    const totalPresses = 4;
    return {
        type: jsPsychHtmlKeyboardResponse,
        choices: "NO_KEYS",
        stimulus: screenHTML({
            title: `Testing cost`,
            lines: [
                `Each time a button is tested, a portion of the room's gold coin <strong>is lost</strong>.`
            ],
            gap: goldCostCoinHTML(),
            stage: `
                <div class="task-row">
                    ${checkButtonHTML({ placeholder: true })}
                    ${initialize_agent()}
                    ${buttonStackHTML()}
                    ${beliefPanelHTML()}
                </div>`,
            below: `<p>Click the buttons a few times to see how this affects the room's gold coin's size.</p>`
        }) + navHTML("Continue"),
        data: { task: "instructions_testing_cost_demo" },
        on_start: function () {
            for (const b of BUTTONS) {
                for (const o of OUTCOMES) counts[b][o] = 0;
            }
            sampleTrueT();
            GOLD_FRACTION = 1;
            agent_topPos = topPos0;
            agent_leftPos = leftPos0;
        },
        on_load: function () {
            refreshBeliefs();
            updateGoldCostCoin();

            const cont = document.getElementById("demo-continue");
            let pressesUsed = 0;

            function armPress() {
                const row = document.querySelector(".task-row");
                const bs = row.querySelector(".button-triangle");
                if (bs) bs.outerHTML = buttonStackHTML();
                const cs = row.querySelector(".check-stack");
                if (cs) cs.outerHTML = checkButtonHTML({ placeholder: true });
                wireButtons(onPress, null);
            }

            function onPress(button) {
                const outcome = sampleCategorical(TRUE_T[button]);
                counts[button][outcome] += 1;
                moveAgent(outcome);

                GOLD_FRACTION = Math.max(0, GOLD_FRACTION - SAMPLE_COST);
                updateGoldCostCoin();

                setTimeout(function () {
                    refreshBeliefs({ button: button, outcome: outcome });
                    setTimeout(function () {
                        const agentEl = document.getElementById("agent");
                        agent_topPos = topPos0;
                        agent_leftPos = leftPos0;
                        agentEl.style.top = topPos0 + "%";
                        agentEl.style.left = leftPos0 + "%";
                        setTimeout(function () {
                            pressesUsed += 1;
                            if (pressesUsed >= totalPresses) {
                                cont.style.display = "inline-block";
                            } else {
                                armPress();
                            }
                        }, MOVE_MS + 150);
                    }, 300);
                }, MOVE_MS);
            }

            armPress();

            cont.addEventListener("click", function () {
                jsPsych.finishTrial({ task: "instructions_testing_cost_demo", presses_used: pressesUsed });
            });
        }
    };
}

// The tick is introduced at a different point depending on whether testing costs
// anything (see make_instructions_timeline), so the slide takes its own lines:
// after the cost demo it is the escape from a depleting coin, after the practice
// it is the alternative to spending a press.
function make_finish_testing_early_demo_trial(opts) {
    opts = opts || {};
    return {
        type: jsPsychHtmlKeyboardResponse,
        choices: "NO_KEYS",
        stimulus: screenHTML({
            title: opts.title || `Finishing testing early`,
            lines: opts.lines || [
                `Although you have the option of testing ${N_REMAINING_TRIALS} more time${N_REMAINING_TRIALS === 1 ? "" : "s"}, you do have the option to <strong>finish testing early</strong>.`,
            ],
            gap: goldCostCoinHTML(),
            stage: `
                <div class="task-row">
                    ${checkButtonHTML({ tick_label: true })}
                    ${initialize_agent()}
                    ${buttonStackHTML()}
                    ${beliefPanelHTML()}
                </div>`,
            below: `<p>Click the tick button to move on to the gold collection phase.</p>`
        }),
        data: { task: "instructions_finish_early_demo" },
        on_start: function () {
            agent_topPos = topPos0;
            agent_leftPos = leftPos0;
        },
        on_load: function () {
            // this slide deliberately keeps the tokens + coin exactly as left by
            // whatever ran before it -- the cost demo when testing costs something,
            // the practice room when it doesn't. counts and GOLD_FRACTION are
            // module-level state, so simply not resetting them here carries the room
            // they were just in straight onto this slide.
            refreshBeliefs();
            updateGoldCostCoin();

            // only the tick is interactive on this slide
            for (const b of BUTTONS) {
                const el = document.getElementById("btn-" + b);
                if (el) el.classList.add("disabled");
            }

            document.getElementById("btn-check").addEventListener("click", function () {
                jsPsych.finishTrial({ task: "instructions_finish_early_demo" });
            });
        }
    };
}

//----------------------------------------------------------------------------//
// Coin-selection trial for instructions.
//   opts.fixedCounts -> set an illustrative tally {red:{...}, blue:{...}} for the
//                 room (overrides the global counts for this demo).
//   opts.useCurrentBeliefs -> keep the tokens already accumulated; the revealed
//                 outcome is sampled from that room's true transition function.
//   opts.goldOutcome -> "maxTokens"/"minTokens" place the coin in the best-/least-
//                 sampled cell; "random"; or a fixed outcome name.
//   opts.reachButton -> that button reaches the coin, the other misses.
//   opts.forceOutcome -> "reach" (any button reaches) / "miss" (any button misses).
//   opts.missFeedback -> feedback text shown on a miss (default "You missed the coin.").
//   opts.buttonOrder -> the room's button layout (default BUTTON_ORDER).
//----------------------------------------------------------------------------//
function make_gold_demo_trial(opts) {
    opts = opts || {};
    const useCurrent = opts.useCurrentBeliefs === true;
    const buttonOrder = opts.buttonOrder || BUTTON_ORDER;
    const introLines = opts.lines || [
        `A gold coin appeared at the <strong>top</strong>.`,
        `Click the button you think is most likely to take you there.`
    ];

    return {
        type: jsPsychHtmlKeyboardResponse,
        choices: "NO_KEYS",
        // laid out as the real gold screen (make_gold_trial): buttons under the room
        stimulus: screenHTML({
            lines: introLines,
            stage: `
                <div class="choice-row">
                    ${roomPanelHTML(initialize_agent_gold(), buttonOrder)}
                </div>`,
            below: opts.note ? `<p class="screen-note">${opts.note}</p>` : ``
        }) + navHTML("Continue"),
        data: { task: useCurrent ? "practice_gold" : "demo_gold" },
        on_start: function () {
            agent_topPos = topPos0;
            agent_leftPos = leftPos0;
            if (opts.fixedCounts) {
                // illustrative room: set the tally from opts.fixedCounts
                for (const b of BUTTONS) {
                    for (const o of OUTCOMES) {
                        counts[b][o] = (opts.fixedCounts[b] && opts.fixedCounts[b][o]) || 0;
                    }
                }
            } else if (!useCurrent) {
                // fallback illustrative beliefs: blue uncertain up/right, red left
                for (const b of BUTTONS) {
                    for (const o of OUTCOMES) counts[b][o] = 0;
                }
                counts.blue.up = 3;
                counts.blue.right = 3;
                counts.red.left = 3;
            }
        },
        on_load: function () {
            let goldOutcome;
            if (opts.goldOutcome === "random") {
                goldOutcome = OUTCOMES[Math.floor(Math.random() * OUTCOMES.length)];
            } else if (opts.goldOutcome === "maxTokens") {
                goldOutcome = outcomeByTokenCount("max");
            } else if (opts.goldOutcome === "minTokens") {
                goldOutcome = outcomeByTokenCount("min");
            } else {
                goldOutcome = opts.goldOutcome || "up";
            }
            placeGold(goldOutcome);
            refreshBeliefs(null, buttonOrder);

            const btnEls = {};
            for (const b of BUTTONS) btnEls[b] = document.getElementById("btn-" + b);
            const cont = document.getElementById("demo-continue");
            const instr = document.getElementById("screen-lines");

            function handle(button) {
                for (const b of BUTTONS) btnEls[b].classList.add("disabled");
                for (const b of BUTTONS) {
                    if (b !== button) btnEls[b].classList.add("hidden");
                }
                instr.innerHTML = ""; // the choice is made; drop the "click a button" prompt

                function missCell() {
                    return OUTCOMES.reduce(function (best, o) {
                        if (o === goldOutcome) return best;
                        return (best === null || counts[button][o] > counts[button][best]) ? o : best;
                    }, null);
                }
                let outcome;
                if (opts.reachButton) {
                    outcome = (button === opts.reachButton) ? goldOutcome : missCell();
                } else if (opts.forceOutcome === "reach") {
                    outcome = goldOutcome;
                } else if (opts.forceOutcome === "miss") {
                    outcome = missCell();
                } else {
                    outcome = useCurrent ? sampleCategorical(TRUE_T[button]) : argmaxOutcome(button);
                }
                const success = outcome === goldOutcome;
                moveAgent(outcome);
                setTimeout(function () {
                    showScreenFeedback(
                        success ? "You reached the gold!" : (opts.missFeedback || "You missed the gold."),
                        success,
                        opts.revealLines
                    );
                    cont.style.display = "inline-block";
                }, MOVE_MS);
            }

            for (const b of BUTTONS) {
                btnEls[b].addEventListener("click", () => handle(b));
            }
            cont.addEventListener("click", () => jsPsych.finishTrial({ task: useCurrent ? "practice_gold" : "demo_gold" }));
        }
    };
}

//----------------------------------------------------------------------------//
// End of instructions: start the experiment, or review the instructions again.
// Sets the global `review_instructions`, which the loop in index.html reads.
//----------------------------------------------------------------------------//
function make_review_choice_trial() {
    return {
        type: jsPsychHtmlButtonResponse,
        stimulus: screenHTML({
            title: `You're ready`,
            lines: [
                `That's the end of the instructions.`,
                `There are <strong>${N_ROOMS} ${N_AFC === 2 ? "pairs" : `sets of ${N_AFC}`} of rooms</strong> to come, each with new buttons to learn.`,
                `We will now ask you a series of questions about the task to check your understanding.`,
                `Before you do so, you have the option to review the instructions again.`
            ]
        }),
        choices: ["Start comprehension check", "Review the instructions again"],
        data: { task: "review_choice" },
        on_finish: function (data) {
            review_instructions = (data.response === 1);
        }
    };
}

//----------------------------------------------------------------------------//
// Build the full instructions timeline (array of trials).
//----------------------------------------------------------------------------//
function instructionBlock(pages, extra) {
    return Object.assign({
        type: jsPsychInstructions,
        show_clickable_nav: true,
        button_label_previous: "Previous",
        button_label_next: "Next",
        data: { task: "instructions" },
        pages: pages
    }, extra || {});
}

function make_instructions_timeline() {
    const tl = [];

    // "<strong>red</strong> and <strong>blue</strong>", or "red, blue and green"
    const bold = BUTTONS.map(b => `<strong>${b}</strong>`);
    const buttonNames = bold.slice(0, -1).join(", ") + " and " + bold[bold.length - 1];
    // what a trial's set of rooms is called: "pairs", or e.g. "sets of 3"
    const roomSets = N_AFC === 2 ? "pairs" : `sets of ${N_AFC}`;

    // Illustrative rooms for the slides explaining the choice, from fixed canonical
    // histories, concretised as loadPresets does a real trial's (one shared colour
    // mapping and button layout, screen order = file order).
    //   testDemo: every room tested, each in a different way -- played out, then
    //             shown as the tally the participant watched build up
    //             (with 3 buttons, the first room tests all three)
    //   untestedDemo: the first room has blue tested far more than red (and green),
    //             the others are not tested at all (canonical action 0 -> blue,
    //             1 -> red, 2 -> green). Location 3 is never reached, which the
    //             gold demos rely on.
    // `buttonPerm` fixes the action -> colour mapping; left out, it is random.
    function demoTrial(histories, buttonPerm) {
        buttonPerm = buttonPerm || shuffled(BUTTONS);
        const order = randomButtonOrder();
        return {
            presets: histories.map(h => buildRoomPreset(h, buttonPerm)),
            orders: histories.map(() => order),
            optionOrder: histories.map((h, j) => j)
        };
    }
    const third = N_BUTTONS === 3;
    const testHistories = [
        [[[0, 0], 3], [[0, 1], 1], [[1, 2], 2]].concat(third ? [[[2, 3], 2]] : []),
        [[[0, 0], 1], [[0, 2], 1], [[1, 1], 1], [[1, 3], 1], [[1, 0], 1]].concat(third ? [[[2, 1], 1]] : []),
        [[[0, 1], 2], [[1, 1], 2], [[1, 3], 1]]
    ];
    const testDemo = demoTrial(testHistories.slice(0, N_AFC));
    const untestedDemo = demoTrial(
        [[[[0, 0], 3], [[0, 1], 1], [[0, 2], 1], [[1, 1], 1]].concat(third ? [[[2, 2], 1]] : [])]
            .concat(Array(N_AFC - 1).fill([])),
        ["blue", "red"].concat(BUTTONS.filter(b => b !== "blue" && b !== "red"))
    );
    // the tested room of untestedDemo, reused by the gold and rating demos
    const exampleRoom = untestedDemo.presets[0];
    const exampleOrder = untestedDemo.orders[0];

    // a demo trial's rooms side by side, as static tallies (`empty` -> no tokens).
    // `activePos`, if given, greys out every room but the one at that position.
    function roomsStaticHTML(demo, empty, activePos) {
        return `
            <div class="choice-row" style="pointer-events:none;">
                ${demo.optionOrder.map((j, pos) => roomPanelHTML(
                    roomCountersStaticHTML(empty ? freshButtonMap() : demo.presets[j].presetCounts, demo.orders[j]),
                    demo.orders[j], { ids: false, idle: activePos !== undefined && pos !== activePos }
                )).join("")}
            </div>`;
    }

    // Two real trials from the presets file for the practice, concretised the same
    // way (here with a shuffled screen order, as in the task). Unlike the real
    // trials, only pairs in which every room has at least one press are used.
    const practiceTrials = shuffled(PRESETS.filter(trial => trial.every(h => h.length > 0))).slice(0, 2).map(function (trial) {
        const demo = demoTrial(trial);
        demo.optionOrder = shuffled(demo.optionOrder);
        return demo;
    });

    // ---- Overview click demo + testing intro + token explanation ----
    tl.push(instructionBlock(function () {
        return [
            screenHTML({
                title: `Button task`,
                lines: [
                `In this experiment, you will encounter a series of rooms.`,
                `There are <strong>${K_OUTCOMES} locations</strong> you can reach from the room's central location - i.e. up, down, left or right.`,
                `Each room has <strong>${N_BUTTONS} buttons</strong> to press, shown below in ${buttonNames}.`,
                `Pressing a button takes you to one of these ${K_OUTCOMES} locations <strong>with some probability</strong>.`,
                ],
                stage: taskDisplayStaticHTML(null, { label_on: false })
            }),
            OBJECTIVE === "gold" ? screenHTML({
                title: `Gold coins`,
                lines: [
                    `At various points in the task, a <strong>gold coin</strong> will appear in one of the ${K_OUTCOMES} reachable locations.`,
                    `The aim of the task is to collect as many gold coins as possible.`,
                    `To do this, you need to press whichever button you think is <strong>most likely</strong> to take you to the gold coin.`,
                    `However, when you first enter each room, <strong>you don't know which locations each button is likely to lead to...</strong>`,
                ],
                stage: taskDisplayStaticHTML(null, { label_on: false, goldOutcome: "up" })
            }) : screenHTML({
                title: `How do the buttons work?`,
                lines: [
                    `When you first enter each room, <strong>you don't know which locations each button is likely to lead to...</strong>`,
                ],
                stage: taskDisplayStaticHTML(null, { label_on: false })
            })
        ];
    }));
    tl.push(make_intro_click_demo_trial());
    tl.push(instructionBlock(function () {
        return [
            screenHTML({
                title: `Tokens`,
                lines: [
                    `Whenever a button is pressed, a <strong>coloured token</strong> will be placed on the location that you reached.`,
                    `Hence, the tokens in each location reflect the <strong>number of times</strong> each button has taken you there.`,
                    `While there is <strong>always some degree of randomness</strong> in where a button leads, buttons can vary in how <strong>reliable</strong> they are.`,
                    `Let's look at ${BUTTONS.length === 2 ? "two" : "three"} examples.`
                ],
                stage: taskDisplayStaticHTML(counts)
            })
        ];
    }));

    // ---- Reliability animations (computer presses each button; Next after 5).
    //      One per button: with 3, reliable (blue) / two directions (red) /
    //      variable (green); with 2, the reliable and the variable demo, the
    //      latter on red. ----
    const tokenDemos = [{
        button: "blue",
        title: "Tokens",
        outcomes: ["up", "up", "up", "right", "up", "up", "left"],
        lines: [
            `For example, a button may have a single preferred direction, reliably leading you to one location.`,
            `See how this <strong>blue</strong> button often takes you <strong>upwards</strong> &mdash; although not all the time.`
        ]
    }];
    if (BUTTONS.length === 3) {
        tokenDemos.push({
            button: "red",
            title: "Tokens",
            reset: false,
            outcomes: ["right", "right", "down", "down", "right", "down", "up", "right",
                "down"
            ],
            lines: [
                `Or, a button may have multiple preferred directions, tending to lead you to those locations, but not others.`,
                `See how this <strong>red</strong> button often takes you either <strong>rightwards</strong> or <strong>downwards</strong> &mdash; although not all the time.`
            ]
        });
    }
    const variableButton = BUTTONS.length === 3 ? "green" : "red";
    tokenDemos.push({
        button: variableButton,
        title: "Tokens",
        reset: false,
        outcomes: ["up", "left", "right", "up", "down", "left", "down", "right"],
        lines: [
            BUTTONS.length === 3
                ? `Other buttons may be much more random, taking you to many different locations.`
                : `Or, a button may be much more random, taking you to many different locations.`,
            `See how this <strong>${variableButton}</strong> button is much more <strong>variable</strong> in the outcomes it leads to.`
        ]
    });
    tokenDemos.forEach(cfg => tl.push(make_auto_demo_trial(cfg)));
    // The demos share the presets' token spacing (COUNTER_SLOTS), so each sends
    // its button to any one location at most 5 times -- the fullest cell in the
    // 3-button presets. Tokens beyond COUNTER_SLOTS would not be drawn.
    tokenDemos.forEach(function (cfg) {
        const most = Math.max.apply(null, OUTCOMES.map(o => cfg.outcomes.filter(x => x === o).length));
        if (most > COUNTER_SLOTS) {
            console.warn(`token demo (${cfg.button}) puts ${most} tokens in one cell, but only ${COUNTER_SLOTS} slots are drawn`);
        }
    });

    // ---- Independence, made on the tokens the three demos just built up.
    //      `pages` is a function so the slide is rendered when it is reached,
    //      off the counts the participant actually saw: the timeline is built
    //      up front (when every count is still zero), and any demo can be
    //      skipped early, so the tally cannot be baked in here. ----
    tl.push(instructionBlock(function () {
        return [
            screenHTML({
                title: `Testing the buttons`,
                lines: [
                    `The ${BUTTONS.length} buttons are <strong>independent</strong>.`,
                    `This means the locations one button tends to reach <strong>may or may not overlap</strong> with those the others tend to reach.`,
                    `Learning about one button therefore tells you nothing about the others.`
                ],
                stage: taskDisplayStaticHTML(counts)
            })
        ];
    }));

    // ---- The cost of testing, only when there is one. With SAMPLE_COST = 0 nothing
    //      depletes, so this slide would teach a rule the task does not have -- and
    //      its coin (goldCostCoinHTML) would render as nothing at all. It sits before
    //      the press budget below, so "here's the catch" lands on a cost the
    //      participant has already been shown. ----
    if (SAMPLE_COST > 0) {
        tl.push(make_testing_cost_demo_trial());
        // straight on, keeping that demo's tokens and part-spent coin on screen:
        // the tick is the way out of the depletion just demonstrated. Without a
        // termination arm there is no way out, and the cost demo stands alone.
        if (TERMINATE) {
            tl.push(make_finish_testing_early_demo_trial({
                lines: [
                    `Luckily, you also have the option to <strong>finish testing early</strong>, by clicking the tick button.`,
                    `Clicking the tick moves you straight on, so no more of the room's gold coin is lost.`
                ]
            }));
        }
    }

    // ---- Each trial is a choice between N_AFC unrelated rooms, whose buttons are
    //      tested for the participant. ----
    tl.push(instructionBlock([
        screenHTML({
            title: `Choosing between rooms`,
            lines: [
                `You will encounter a series of ${roomSets} of rooms, each containing <strong>${N_BUTTONS} buttons</strong>.`,
                `These rooms are <strong>unrelated</strong> to one another &mdash; how the buttons work in one room has nothing to do with how they work in the other room.`
            ],
            stage: roomsStaticHTML(testDemo, true)
        }),
        screenHTML({
            title: `Choosing between rooms`,
            lines: [
                `In fact, when you are shown these ${N_AFC === 2 ? "two" : N_AFC} rooms, the buttons in each will be <strong>tested automatically</strong> &mdash; that is, they will be pressed for you, and you will observe their outcomes.`,
                `Click Next to watch the rooms being tested.`
            ],
            stage: roomsStaticHTML(testDemo, true)
        })
    ]));
    // choice_num -3..-1 keep these rows out of the way of the real trials (1..N_ROOMS)
    tl.push(make_room_choice(-3, Object.assign({ practice: true, watchOnly: true }, testDemo)));

    // ---- How much each room is tested varies: the pair just watched, then a
    //      room testing blue far more than red (the empty room greyed out), then
    //      the untested room (the tested one greyed out) ----
    const objectivePages = [
        screenHTML({
            title: `Which buttons are tested in each room?`,
            lines: [
                `The number of times the buttons are pressed for you before you select your room, and which buttons are pressed, <strong>varies between trials</strong>.`
            ],
            stage: roomsStaticHTML(testDemo)
        }),
        screenHTML({
            title: `Which buttons are tested in each room?`,
            lines: [
                `For example, it may be the case that in one of the rooms, the <strong>blue</strong> button is tested far more times than the ${third ? `<strong>red</strong> and <strong>green</strong> buttons` : `<strong>red</strong> button`}.`
            ],
            stage: roomsStaticHTML(untestedDemo, false, 0)
        }),
        screenHTML({
            title: `Which buttons are tested in each room?`,
            lines: [
                `Or, it may even be that the buttons in one of the rooms in the ${N_AFC === 2 ? "pair" : "set"} are <strong>not tested at all</strong>.`
            ],
            stage: roomsStaticHTML(untestedDemo, false, 1)
        }),
        // ---- The objective, on the same pair of rooms ----
        screenHTML({
            title: `Choosing between rooms`,
            lines: [
                OBJECTIVE === "gold"
                    ? `Your job is to select the room in which you think you have the <strong>best chance of obtaining the gold coin</strong> when it appears in one of the locations.`
                    : `Your job is to select the room in which you feel the <strong>strongest sense of control</strong>.`,
                `You can select a room by clicking on it.`
            ],
            stage: roomsStaticHTML(untestedDemo)
        })
    ];
    if (OBJECTIVE === "control") {
        objectivePages.push(screenHTML({
            title: `What does it mean to feel a sense of control?`,
            lines: [
                `Well, this is exactly what we are trying to find out!`,
                `Different people might feel in control in different ways. For example, someone might feel like the room on the left gives them more control, while someone might prefer the room on the right.`
            ],
            stage: roomsStaticHTML(untestedDemo)
        }));
        objectivePages.push(screenHTML({
            title: `What does it mean to feel a sense of control?`,
            lines: [
                `There is <strong>no correct answer</strong>, so please choose as honestly as you can depending on your understanding of what it means to be in control.`
            ],
            stage: roomsStaticHTML(untestedDemo)
        }));
    }
    tl.push(instructionBlock(objectivePages));

    // ---- The rating that follows the choice, tried out on one of the rooms. Under
    //      "gold" it comes after the gold-collection demos below. ----
    function pushRatingDemo() {
        if (!CONFIDENCE) return;
        tl.push(make_confidence_trial(-3, {
            practice: true,
            preset: exampleRoom,
            buttonOrder: exampleOrder,
            title: `Rating your choice`,
            question: false,
            lines: [
                OBJECTIVE === "gold"
                    ? `After you select one of the rooms, you will then be asked to rate <strong>how likely you think you are to reach the gold coin</strong> in this room.`
                    : `After you select one of the rooms, you will then be asked to rate <strong>how strongly you feel a sense of control</strong> in this room.`,
                `To do so, please adjust the slider below.`
            ]
        }));
    }
    if (OBJECTIVE !== "gold") pushRatingDemo();

    // ---- The gold-collection phase (gold objective only), in the tested room of
    //      the pair just shown: blue mostly reaches canonical location 0, and
    //      location 3 is never reached by any button -- used to script the two
    //      coin demos. ----
    if (OBJECTIVE === "gold") {
        const goldDemoCounts = exampleRoom.presetCounts;
        const blueFavourite = exampleRoom.outcome_map[0];
        const unreached = exampleRoom.outcome_map[3];
        tl.push(instructionBlock([
            screenHTML({
                title: `Collecting the gold`,
                lines: [
                    `Once you've selected a room, a <strong>gold coin</strong> will appear at one of the ${K_OUTCOMES} locations in that room.`,
                    `You must then choose the button you think is <strong>most likely to take you to the gold</strong>, based on what you have learned about each button.`,
                    `Let's try a couple of examples.`
                ],
                stage: goldCoinStaticHTML()
            })
        ]));

        // ---- Coin-selection demo 1: coin where blue mostly goes. Blue reaches it
        //      (success message); the others do not. ----
        tl.push(make_gold_demo_trial({
            fixedCounts: goldDemoCounts,
            buttonOrder: exampleOrder,
            goldOutcome: blueFavourite,
            reachButton: "blue", // blue reaches the coin; the other buttons miss
            missFeedback: "You missed the coin.",
            lines: [
                `A <strong>gold coin</strong> has appeared.`,
                `Click the button you think is most likely to reach it.`
            ],
            note: `(Note that in the real experiment, you will not see whether or not you actually reached the gold.)`
        }));

        // ---- Coin-selection demo 2: coin where no button has gone. ----
        tl.push(make_gold_demo_trial({
            fixedCounts: goldDemoCounts,
            buttonOrder: exampleOrder,
            goldOutcome: unreached,
            forceOutcome: "miss", // no button reaches this coin
            lines: [
                `Here's another gold coin.`,
                `Again, click the button you think is most likely to reach it.`
            ],
            note: `(Note that in the real experiment, you will not see whether or not you actually reached the gold.)`,
            revealLines: [
                `There's <strong>no guarantee</strong> the gold coin will appear somewhere any button can reliably reach.`,
                `The best you can do is to choose the button you think is most likely to reach the gold coin`
            ]
        }));

        pushRatingDemo();
    }

    // ---- Two practice trials, run exactly as the real ones ----
    tl.push(instructionBlock([
        screenHTML({
            title: `Let's practise`,
            lines: [`Let's do a couple of practice trials.`]
        })
    ]));
    practiceTrials.forEach(function (trial, i) {
        const n = i - 2;   // -2, -1
        if (i > 0) {
            tl.push({
                type: jsPsychHtmlKeyboardResponse,
                stimulus: screenHTML({
                    title: `Practice trial ${i + 1}`,
                    lines: [`Press any key to continue to the next practice trial.`]
                }),
                data: { task: "practice_break" }
            });
        }
        tl.push(make_room_choice(n, Object.assign({ practice: true }, trial)));
        if (CONFIDENCE) tl.push(make_confidence_trial(n, { practice: true }));
        if (OBJECTIVE === "gold") {
            tl.push(make_gold_pause(n));
            tl.push(make_gold_trial(n, { practice: true }));
        }
    });

    // ---- The aim of the task ----
    tl.push(instructionBlock([
        screenHTML({
            title: `The aim of the task`,
            lines: OBJECTIVE === "gold" ? [
                `The aim of the task is to collect <strong>as many gold coins as possible</strong>.`,
                `The more gold coins you collect, the <strong>bigger the bonus</strong> you will receive on Prolific.`,
                `So, make sure you select whichever room you think gives you the <strong>best chance of reaching the gold</strong>, given what you have learned about how the buttons work.`
            ] : [
                `There are <strong>no right or wrong answers</strong>, and there is no bonus for choosing particular rooms.`,
                `Please choose the room in which you genuinely feel the <strong>strongest sense of control</strong>, and answer as honestly as you can.`,
                `We are very grateful for your participation, and for helping us to understand what it means for people to feel a <strong>sense of control</strong>.`
            ]
        })
    ]));

    // ---- End: start, or review the instructions again ----
    tl.push(make_review_choice_trial());


    return tl;
}
