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
    const revealAfter = cfg.revealAfter || 1;
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
            title: `Testing phase`,
            lines: [
                `Fortunately, before the gold appears, you can test the buttons to learn how they work.`,
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
//----------------------------------------------------------------------------//
function make_gold_demo_trial(opts) {
    opts = opts || {};
    const useCurrent = opts.useCurrentBeliefs === true;
    const introLines = opts.lines || [
        `A gold coin appeared at the <strong>top</strong>.`,
        `Click the button you think is most likely to take you there.`
    ];

    return {
        type: jsPsychHtmlKeyboardResponse,
        choices: "NO_KEYS",
        stimulus: screenHTML({
            lines: introLines,
            stage: `
                <div class="task-row">
                    ${checkButtonHTML({ placeholder: true })}
                    ${initialize_agent_gold()}
                    ${buttonStackHTML()}
                    ${beliefPanelHTML()}
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
            refreshBeliefs();

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
                `There are <strong>${N_ROOMS} rooms</strong> to come, each with new buttons to learn.`,
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
// Prime the globals a practice room needs, exactly as make_room_intro does for a
// real one: an empty tally, a full coin, and hidden dynamics drawn from the
// posterior implied by the history that is about to be watched -- so the presses
// the participant watches are consistent with what their own press then does.
//----------------------------------------------------------------------------//
function startPracticeRoom(preset) {
    for (const b of BUTTONS) {
        for (const o of OUTCOMES) counts[b][o] = 0;
    }
    GOLD_FRACTION = 1;
    sampling_ended = false;
    agent_topPos = topPos0;
    agent_leftPos = leftPos0;
    sampleTrueTFromPreset(preset.presetCounts);
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

    // How many presses of their own the participant gets in a room, and whether
    // that reads as "1 time" or "3 times". Used throughout the slides below.
    const nOwn = N_REMAINING_TRIALS;
    const sOwn = nOwn === 1 ? "" : "s";

    // Two real preset histories, drawn at random and concretised the same way a
    // room's is (buildRoomPreset gives each its own colour/direction mapping and
    // shuffles the presses). One for the practice that teaches the structure, one
    // for the full practice room at the end -- different histories, so the second
    // is not a replay of the first.
    const practicePresets = shuffled(PRESETS).slice(0, 2).map(buildRoomPreset);

    // ---- Overview click demo + testing intro + token explanation ----
    tl.push(instructionBlock(function () {
        return [
            screenHTML({
                title: `Button task`,
                lines: [
                `In this experiment, you will encounter a series of rooms, one at a time.`,
                `There are <strong>${K_OUTCOMES} locations</strong> you can reach from the room's central location - i.e. up, down, left or right.`,
                `Each room has <strong>${N_BUTTONS} buttons</strong> to press, shown below in <strong>${BUTTONS[0]}</strong>, <strong>${BUTTONS[1]}</strong> and <strong>${BUTTONS[2]}</strong>.`,
                `Pressing a button takes you to one of these ${K_OUTCOMES} locations <strong>with some probability</strong>.`,
                ],
                stage: taskDisplayStaticHTML(null, { label_on: false })
            }),
            screenHTML({
                title: `Gold coins`,
                lines: [
                    `At various points in the task, a <strong>gold coin</strong> will appear in one of the ${K_OUTCOMES} reachable locations.`,
                    `The aim of the task is to collect as many gold coins as possible.`,
                    `To do this, you need to press whichever button you think is <strong>most likely</strong> to take you to the gold coin.`,
                    `However, when you first enter each room, <strong>you don't know which locations each button is likely to lead to...</strong>`,
                ],
                stage: taskDisplayStaticHTML(null, { label_on: false, goldOutcome: "up" })
            })
        ];
    }));
    tl.push(make_intro_click_demo_trial());
    tl.push(instructionBlock(function () {
        return [
            screenHTML({
                title: `Tokens`,
                lines: [
                    `Whenever you press a button, a <strong>coloured token</strong> will be placed on the location that you reached.`,
                    `Hence, the tokens in each location reflect the <strong>number of times</strong> each button has taken you there.`,
                    `While there is <strong>always some degree of randomness</strong> in where a button leads, buttons can vary in how <strong>reliable</strong> they are.`,
                    `Let's look at three examples.`
                ],
                stage: taskDisplayStaticHTML(counts)
            })
        ];
    }));

    // ---- Reliability animations (computer presses each button; Next after 5) ----
    tl.push(make_auto_demo_trial({
        button: "blue",
        title: "Tokens",
        outcomes: ["up", "up", "up", "right", "up", "up", "left", "up",
            "up", "up"
        ],
        lines: [
            `For example, a button may have a single preferred direction, reliably leading you to one location.`,
            `See how this <strong>blue</strong> button often takes you <strong>upwards</strong> &mdash; although not all the time.`
        ]
    }));
    tl.push(make_auto_demo_trial({
        button: "red",
        title: "Tokens",
        reset: false, 
        outcomes: ["right", "right", "down", "down", "right", "down", "up", "right",
            "down", "down"
        ],
        lines: [
            `Or, a button may have multiple preferred directions, tending to lead you to those locations, but not others.`,
            `See how this <strong>red</strong> button often takes you either <strong>rightwards</strong> or <strong>downwards</strong> &mdash; although not all the time.`
        ]
    }));
    tl.push(make_auto_demo_trial({
        button: "green",
        title: "Tokens",
        reset: false, 
        outcomes: ["up", "left", "right", "up", "down", "left", "down", "down",
            "up", "left"
        ],
        lines: [
            `Other buttons may be much more random, taking you to many different locations.`,
            `See how this <strong>green</strong> button is much more <strong>variable</strong> in the outcomes it leads to.`
        ]
    }));

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
        // the tick is the way out of the depletion just demonstrated
        tl.push(make_finish_testing_early_demo_trial({
            lines: [
                `Luckily, you also have the option to <strong>finish testing early</strong>, by clicking the tick button.`,
                `Clicking the tick moves you straight on, so no more of the room's gold coin is lost.`
            ]
        }));
    }

    // ---- The press budget: a history watched, then N_REMAINING_TRIALS presses of
    //      their own. Explained, then practised immediately. ----
    tl.push(instructionBlock([
        screenHTML({
            title: `How do I test the buttons?`,
            lines: [
                `Here's the catch: when you enter a room, the buttons will first be tested automatically &mdash; that is, the buttons will be pressed for you, and you will observe their outcomes.`,
                `Once these automatic presses are complete, you will then have the chance to press any of the buttons <strong>${nOwn} more time${sOwn}</strong> yourself.`,
                `Let's practise...`
            ],
            // the room the practice is about to start in: the observation phase
            // opens on exactly this picture, so the buttons just begin pressing
            stage: taskDisplayStaticHTML()
        })
    ], {
        on_start: function () { startPracticeRoom(practicePresets[0]); }
    }));
    // room_num -1 keeps these rows out of the way of the full practice room (0) and
    // of the real rooms (1..N_ROOMS)
    tl.push(make_room_demo(-1, { preset: practicePresets[0], practice: true }));
    // the tick is only offered here if it has already been explained (which it has,
    // above, when testing costs something); otherwise it appears for the first time
    // on the slide that introduces it, just below
    tl.push(make_room_sampling(-1, {
        practice: true,
        nTrials: N_REMAINING_TRIALS,
        tick: SAMPLE_COST > 0
    }));
    
    // ---- The point of the observation phase, on the room they have just left. ----
    tl.push(instructionBlock(function () {
        return [
            screenHTML({
                title: `How do I test the buttons?`,
                lines: [
                    `The number of times the buttons are pressed for you, and which buttons are pressed, varies from room to room.`,
                    `So, this means you will already have some knowledge about how the buttons work before you start pressing them yourself.`
                ],
                stage: taskDisplayStaticHTML(counts)
            })
        ];
    }));

    if (SAMPLE_COST === 0) {
        tl.push(make_finish_testing_early_demo_trial());
    }

    // ---- The gold-collection phase. A fresh illustrative room (unrelated to the
    //      practice): red reaches "up" reliably, blue tends "right", and "down" is
    //      never reached by either -- used to script the two coin demos. ----
    const goldDemoCounts = {
        red:  { up: 2, right: 0, down: 0, left: 0 },
        blue: { up: 0, right: 1, down: 1, left: 0 }
    };
    // a representative four-press depletion, so these slides show the coin as a
    // tested room leaves it. Whole when testing is free.
    const demoCoinFraction = Math.max(0, 1 - 4 * SAMPLE_COST);
    tl.push(instructionBlock([
        screenHTML({
            title: `Collecting the gold`,
            lines: [
                SAMPLE_COST > 0
                    ? `Once you've finished testing the buttons, whatever is left of the room's <strong>gold coin</strong> will appear at one of the ${K_OUTCOMES} locations.`
                    : `Once you've finished testing the buttons, the room's <strong>gold coin</strong> will appear at one of the ${K_OUTCOMES} locations.`,
                `You must then choose the button you think is <strong>most likely to take you to the gold</strong>, based on what you have learned about each button.`
            ].concat(SAMPLE_COST > 0 ? [
                `Note that the size of the gold coin, and therefore its value, is determined by how many times you just tested the buttons.`
            ] : []).concat([
                `Here's a fresh room. Let's try a couple of examples.`
            ]),
            stage: goldCoinStaticHTML(demoCoinFraction)
        })
    ]));

    // ---- Coin-selection demo 1: coin at "up", where red is reliable. Red reaches
    //      it (success message); blue does not. ----
    tl.push(make_gold_demo_trial({
        fixedCounts: goldDemoCounts,
        goldOutcome: "up",
        reachButton: "red", // red reaches the coin; the other button misses
        missFeedback: "You missed the coin.",
        lines: [
            `A <strong>gold coin</strong> has appeared.`,
            `Click the button you think is most likely to reach it.`
        ],
        note: `(Note that in the real experiment, you will not see whether or not you actually reached the gold.)`
    }));

    // ---- Coin-selection demo 2: coin at "down", which neither button reaches. ----
    tl.push(make_gold_demo_trial({
        fixedCounts: goldDemoCounts,
        goldOutcome: "down",
        forceOutcome: "miss", // neither button reaches this coin
        lines: [
            `Here's another gold coin.`,
            `Again, click the button you think is most likely to reach it.`
        ],
        note: `(Note that in the real experiment, you will not see whether or not you actually reached the gold.)`,
        revealLines: [
            `There's <strong>no guarantee</strong> the gold coin will appear somewhere any button can reliably reach.`,
            `Sometimes it might simply be unlikely that you will get it.`,
            `The best you can do is to choose the button you think is most likely to reach the gold coin`
        ]
    }));

    // ---- The trade-off only exists when testing costs something. ----
    if (SAMPLE_COST > 0) {
        tl.push(instructionBlock([
            screenHTML({
                title: `Balancing testing vs gold collection`,
                lines: [
                    `Remember: if you manage to reach the gold coin, its value is determined by its size.`,
                    `Therefore, you are faced with a trade-off:`,
                    `<strong>- Testing the buttons in order to learn enough about how they work...</strong>`,
                    `<strong>- ...while also balancing this against the cost of testing.</strong>`
                ],
                stage: goldCoinStaticHTML(demoCoinFraction)
            })
        ]));
    }

    // ---- A full room, gold phase included, on a second preset history. ----
    tl.push(instructionBlock([
        screenHTML({
            title: `Let's practise`,
            lines: [
                `The next part works exactly like a real room.`,
                `The buttons will be pressed for you first, and then you have the chance to test <strong>${nOwn} more time${sOwn}</strong>.`,
                `Feel free to use your ${nOwn} extra choice${sOwn}, or click the tick button when you feel you've learned enough.`
            ]
        })
    ], {
        on_start: function () { startPracticeRoom(practicePresets[1]); }
    }));

    tl.push(make_room_demo(0, { preset: practicePresets[1], practice: true }));
    tl.push(make_room_sampling(0, { practice: true, nTrials: N_REMAINING_TRIALS }));
    tl.push(make_gold_pause(0));
    tl.push(make_gold_trial(0, { practice: true }));

    // ---- The aim of the task (bonus) + note that the real task hides the outcome ----
    tl.push(instructionBlock([
        screenHTML({
            title: `The aim of the task`,
            lines: SAMPLE_COST > 0 ? [
                `The aim of the task is to maximise the <strong>combined value of the gold coins</strong> that you collect.`,
                `The more gold coins you collect, and the more valuable they are, the <strong>bigger the bonus</strong> you will receive on Prolific.`,
                `<strong>Remember:</strong> the more times you test the buttons in a room, the smaller and less valuable the gold coin that will appear in that room.`,
                `So in each room, test out the buttons until you feel you've learned enough to continue to the gold collection phase.`,
            ] : [
                `The aim of the task is to collect <strong>as many gold coins as possible</strong>.`,
                `The more gold coins you collect, the <strong>bigger the bonus</strong> you will receive on Prolific.`,
                `So in each room, use what you have watched &mdash; and your own ${nOwn} choice${sOwn} &mdash; to work out which button is most likely to reach the gold.`,
            ]
        })
    ]));

    // ---- End: start, or review the instructions again ----
    tl.push(make_review_choice_trial());


    return tl;
}
