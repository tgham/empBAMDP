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
function reachableRoomStaticHTML() {
    let cells = "";
    // for (const outcome of OUTCOMES) {
    //     cells += `<div class="reachable-cell" style="${cellPosStyle(OUTCOME_CELL[outcome], 0.08, 0.84)}"></div>`;
    // }
    return `
        <div class="container">
            <img src="img/BaseAction_4k.png" alt="Base" class="base-image">
            <div class="belief-overlay">${cells}</div>
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

// a full static illustration: room + buttons + belief display, for given counts.
// Adapts to BELIEF_DISPLAY (counters -> single grid of tokens; else two heatmaps).
function exampleDisplayStaticHTML(redCounts, blueCounts) {
    if (BELIEF_DISPLAY === "counters") {
        return `
            <div class="task-row" style="pointer-events:none;">
                ${roomCountersStaticHTML(redCounts, blueCounts)}
                ${buttonStackHTML()}
            </div>`;
    }
    return `
        <div class="task-row" style="pointer-events:none;">
            ${reachableRoomStaticHTML()}
            ${buttonStackHTML()}
            <div class="belief-stack">
                ${beliefBlockStaticHTML("blue", "Blue button", blueCounts)}
                ${beliefBlockStaticHTML("red", "Red button", redCounts)}
            </div>
        </div>`;
}

// room + buttons only (no heatmaps), for the overview slides
function roomButtonsStaticHTML() {
    return `
        <div class="task-row" style="pointer-events:none;">
            ${reachableRoomStaticHTML()}
            ${buttonStackHTML()}
        </div>`;
}

// the plain room (base + centred agent, no highlights) for realistic illustrations
function plainRoomStaticHTML() {
    return `
        <div class="container">
            <img src="img/BaseAction_4k.png" alt="Base" class="base-image">
            <img src="img/Agent.png" alt="Agent" class="agent-image">
        </div>`;
}

// the usual task display (tick + room + buttons + belief display), illustrating
// early stop. Adapts to BELIEF_DISPLAY.
function earlyStopStaticHTML() {
    // blank grid (no tokens) for the "Testing the buttons" slide
    const blueC = { up: 0, right: 0, down: 0, left: 0 };
    const redC = { up: 0, right: 0, down: 0, left: 0 };
    if (BELIEF_DISPLAY === "counters") {
        return `
            <div class="task-row" style="pointer-events:none;">
                ${checkButtonHTML()}
                ${roomCountersStaticHTML(redC, blueC)}
                ${buttonStackHTML()}
            </div>`;
    }
    return `
        <div class="task-row" style="pointer-events:none;">
            ${checkButtonHTML()}
            ${plainRoomStaticHTML()}
            ${buttonStackHTML()}
            <div class="belief-stack">
                ${beliefBlockStaticHTML("blue", "Blue button", blueC)}
                ${beliefBlockStaticHTML("red", "Red button", redC)}
            </div>
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
        const total = counts.red[outcome] + counts.blue[outcome];
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
        stimulus: `
            ${cfg.title ? `<h2 style="margin-top:0;">${cfg.title}</h2>` : ``}
            <div class="task-row">
                ${checkButtonHTML({ placeholder: true })}
                ${initialize_agent()}
                ${buttonStackHTML()}
                ${beliefPanelHTML()}
            </div>
            <div class="prompt">
                <div id="demo-instruction" class="demo-text">${cfg.instruction}</div>
                <button id="demo-continue" class="demo-continue" style="display:none;">Next</button>
            </div>`,
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
            refreshBeliefs(); // empty tally to begin with

            const active = document.getElementById("btn-" + cfg.button);
            const other = document.getElementById("btn-" + (cfg.button === "red" ? "blue" : "red"));
            // buttons are computer-controlled here; fade the one not being demoed
            active.classList.add("inert");
            other.classList.add("inert");
            other.style.opacity = "0.2";

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
    const introText = opts.instruction ||
        `A gold coin appeared at the <strong>top</strong>.<br>` +
        `Click the button you think is most likely to take you there.`;

    return {
        type: jsPsychHtmlKeyboardResponse,
        choices: "NO_KEYS",
        stimulus: `
            <div class="task-row">
                ${checkButtonHTML({ placeholder: true })}
                ${initialize_agent_gold()}
                ${buttonStackHTML()}
                ${beliefPanelHTML()}
            </div>
            <div class="prompt">
                <div id="demo-instruction" class="demo-text">${introText}</div>
                <h2 id="gold-feedback"></h2>
                ${opts.note ? `<div id="demo-note" class="demo-text">${opts.note}</div>` : ``}
                <button id="demo-continue" class="demo-continue" style="display:none;">Continue</button>
            </div>`,
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

            const btnRed = document.getElementById("btn-red");
            const btnBlue = document.getElementById("btn-blue");
            const cont = document.getElementById("demo-continue");
            const instr = document.getElementById("demo-instruction");

            function handle(button) {
                btnRed.classList.add("disabled");
                btnBlue.classList.add("disabled");
                (button === "red" ? btnBlue : btnRed).classList.add("hidden");

                // the demos always reveal the outcome (for teaching), even though the
                // real experiment keeps it hidden (SHOW_GOLD_OUTCOME).
                // Scripted results: reachButton -> that button reaches the coin and
                // the other misses; forceOutcome "reach"/"miss" ignores the button.
                // A "miss" moves to the button's most-sampled non-coin cell.
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
                    const fb = document.getElementById("gold-feedback");
                    if (success) { fb.textContent = "You reached the coin!"; fb.style.color = "green"; }
                    else { fb.textContent = opts.missFeedback || "You missed the coin."; fb.style.color = "red"; }
                    instr.innerHTML = opts.revealText || `Click <strong>Continue</strong>.`;
                    cont.style.display = "inline-block";
                }, MOVE_MS);
            }

            btnRed.addEventListener("click", () => handle("red"));
            btnBlue.addEventListener("click", () => handle("blue"));
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
        stimulus: `
            <h2>You're ready</h2>
            <p style="max-width:640px; margin:0 auto;">That's the end of the instructions.</p>
            <p style="max-width:640px; margin:0 auto;">There are <strong>${N_ROOMS} rooms</strong> to come, each with new buttons to learn.</p>
            <p style="max-width:640px; margin:0 auto;">Would you like to start, or review the instructions again from the beginning?</p>`,
        choices: ["Start the experiment", "Review the instructions again"],
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

// join sentences with a blank line between them (readable spacing in a prompt)
function sayLines() {
    return Array.prototype.slice.call(arguments).join("<br><br>");
}

function make_instructions_timeline() {
    const tl = [];
    const zeroCounts = { up: 0, right: 0, down: 0, left: 0 };
    const P = `max-width:680px; margin:14px auto;`; // paragraph style, generous spacing

    // ---- Overview + testing intro + token explanation (before the demos) ----
    tl.push(instructionBlock([
        `<h2>Button task</h2>
         <p style="${P}">In the next phase you will explore a series of rooms, one at a time.</p>
         <p style="${P}">In each room you have <strong>${N_BUTTONS} coloured buttons</strong> to press, and there are
         <strong>${K_OUTCOMES} locations</strong> you can reach - i.e. up, down, left or right.</p>
         <p style="${P}">Each button takes you to one of these ${K_OUTCOMES} locations, but
         <strong>you don't know which location each button is most likely to lead to</strong>.</p>
         <p style="${P}">NOTE: the colour of the buttons has <strong>no relation</strong> to the locations they reach.</p>
         ${roomButtonsStaticHTML()}`,

        `<h2>Testing the buttons</h2>
        <p style="${P}">Whenever you press a button, a <strong>coloured token</strong> will be placed on the location
        that you reached.</p>
        <p style="${P}">Hence, the tokens in each location reflect the <strong>number of times</strong> each button has
        taken you there.</p>
        <p style="${P}">While there is <strong>always some degree of randomness</strong> in where a button leads,
        buttons can vary in how <strong>reliable</strong> they are.</p>
         <p style="${P}">Let's look at two examples.</p>
         ${exampleDisplayStaticHTML(zeroCounts, zeroCounts)}`
    ]));

    // ---- Reliability animations (computer presses each button; Next after 5) ----
    tl.push(make_auto_demo_trial({
        button: "blue",
        title: "Testing the buttons",
        outcomes: ["up", "up", "up", "right", "up", "up", "left", "up",
            // "up", "up"
        ],
        instruction: sayLines(
            `For example, a button may have a preferred direction, reliably leading you to one location.`,
            `See how this <strong>blue</strong> button often takes you <strong>upwards</strong> &mdash; although not all the time.`
        )
    }));
    tl.push(make_auto_demo_trial({
        button: "red",
        title: "Testing the buttons",
        reset: false, // keep the blue tokens from the first demo, to show independence
        outcomes: ["up", "left", "right", "up", "down", "left", "down", "down",
            // "up", "left"
        ],
        instruction: sayLines(
            `Other buttons may be more random, taking you to many different locations.`,
            `See how this <strong>red</strong> button is much more variable in the outcomes it leads to.`,
            `The two buttons are <strong>independent</strong>. This means the locations one button tends to reach
            may or may not overlap with those the other reaches.`,
            `Learning about one button therefore tells you nothing about the other.`,
        )
    }));

    // ---- Interpreting the tokens: blank grid at first, then how tokens in one
    //      location make the others less likely. ----
    tl.push(instructionBlock([
        `<h2>Interpreting the tokens</h2>
         <p style="${P}">At the beginning, before a button has been tested, there are <strong>no tokens</strong>.
         So it's reasonable to believe each button is just as likely to lead <strong>anywhere</strong>.</p>
         ${exampleDisplayStaticHTML(zeroCounts, zeroCounts)}`,

        `<h2>Interpreting the tokens</h2>
         <p style="${P}">If, however, testing a button adds more tokens to one location, this indicates the button is 
         <strong>less</strong> likely to lead to the others.</p>
         <p style="${P}">For example, in the room below the tokens suggest the <strong>blue</strong> button is likely
         to lead <strong>up</strong> &mdash; suggesting it is less likely to lead to any of the other locations.</p>
         ${exampleDisplayStaticHTML({ up: 0, right: 0, down: 0, left: 0 }, { up: 5, right: 0, down: 0, left: 0 })}`
    ]));

    // (No single-selection practice here: participants practise a full room below.)

    // ---- Testing the buttons: pressing, budget, and the tick to finish early.
    //      Full task display beneath. on_start sets up the practice room that
    //      follows (also runs on review, before the conditional practice trials). ----
    tl.push(instructionBlock([
        `<h2>Testing the buttons</h2>
         <p style="${P}">You can press a button by clicking it, and then observing where it took you.</p>
         <p style="${P}">You can test the buttons up to <strong>${N_TRIALS} times</strong> in total, splitting your
         presses between the buttons however you like.</p>
         <p style="${P}">You do not have to use all ${N_TRIALS} choices, though &mdash; if you already feel certain
         enough, you can click the <strong>tick button</strong> to move on to the next phase.</p>
         ${earlyStopStaticHTML()}`
    ], {
        on_start: function () {
            for (const b of BUTTONS) {
                for (const o of OUTCOMES) counts[b][o] = 0;
            }
            sampleTrueT();
            sampling_ended = false;
        }
    }));

    // ---- Practice: press the buttons (up to N_TRIALS) until certain, then tick ----
    tl.push(instructionBlock([
        `<h2>Let's practise</h2>
         <p style="${P}">The next part works exactly like a real room.</p>
         <p style="${P}">Press the buttons to test them, and click the tick button when you feel
         you've learned enough.</p>`
    ]));
    tl.push(make_room_sampling(0, { practice: true }));

    // ---- The gold-collection phase. A fresh illustrative room (unrelated to the
    //      practice): red reaches "up" reliably, blue tends "right", and "down" is
    //      never reached by either -- used to script the two coin demos. ----
    const goldDemoCounts = {
        red:  { up: 5, right: 0, down: 0, left: 0 },
        blue: { up: 0, right: 2, down: 0, left: 1 }
    };
    tl.push(instructionBlock([
        `<h2>Collecting the gold</h2>
         <p style="${P}">Once you've finished testing, a <strong>gold coin</strong> will appear at one of the
         ${K_OUTCOMES} locations.</p>
         <p style="${P}">You must then choose the <strong>button you think is most likely to take you to the
         coin</strong>, based on the tokens you've collected for each button.</p>
         <p style="${P}">Here's a fresh room. Let's try a couple of examples.</p>`
    ]));

    // ---- Coin-selection demo 1: coin at "up", where red is reliable. Red reaches
    //      it (success message); blue does not. ----
    tl.push(make_gold_demo_trial({
        fixedCounts: goldDemoCounts,
        goldOutcome: "up",
        reachButton: "red", // red reaches the coin; the other button misses
        missFeedback: "The button you selected didn't take you to the coin this time.",
        instruction: sayLines(
            `A <strong>gold coin</strong> has appeared.`,
            `Click the button you think is most likely to reach it.`
        ),
        note: `<em style="font-size:0.85em; color:#555;">(Note that in the real experiment, you will not see whether or not you actually reached the coin.)</em>`,
    }));

    // ---- Coin-selection demo 2: coin at "down", which neither button reaches. ----
    tl.push(make_gold_demo_trial({
        fixedCounts: goldDemoCounts,
        goldOutcome: "down",
        forceOutcome: "miss", // neither button reaches this coin
        instruction: sayLines(
            `Here's another coin.`,
            `Again, click the button you think is most likely to reach it.`
        ),
        note: `<em style="font-size:0.85em; color:#555;">(Note that in the real experiment, you will not see whether or not you actually reached the coin.)</em>`,
        revealText: sayLines(
            `There's no guarantee the coin will appear somewhere a button can reliably reach &mdash; sometimes it might simply be unlikely that you will get it.`
        )
    }));

    // ---- The aim of the task (bonus) + note that the real task hides the outcome ----
    tl.push(instructionBlock([
        `<h2>The aim of the task</h2>
         <p style="${P}">The aim of the task is to collect <strong>as many gold coins as possible</strong>.</p>
         <p style="${P}">The more coins you collect, the <strong>bigger the bonus</strong> you will receive on Prolific.</p>
         <p style="${P}">So in each room, test out the buttons until you feel <strong>certain enough</strong> to continue to the
         gold selection phase.</p>
         <p style="${P}">Remember: in the practice we <strong>showed you whether you reached the coin</strong>.
         In the real experiment you will <strong>not</strong> see this, so just choose the button you
         most believe will take you to the coin.</p>`
    ]));

    // ---- End: start, or review the instructions again ----
    tl.push(make_review_choice_trial());

    return tl;
}
