//----------------------------------------------------------------------------//
// Instructions phase: slide-based text (jsPsychInstructions, next/prev) with two
// interactive demo trials interleaved so participants practise clicking a button
// and watch the heatmap update (and weaken when a different outcome is observed).
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

// the main room with a gold coin at one location (illustration)
function goldRoomStaticHTML(outcome) {
    const { top, left } = outcomePercent(outcome);
    return `
        <div class="container">
            <img src="img/BaseAction_4k.png" alt="Base" class="base-image">
            <img src="img/Goal.png" alt="Gold" class="gold-image" style="top:${top}%; left:${left}%;">
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
    const blueC = { up: 2, right: 0, down: 0, left: 1 };
    const redC = { up: 0, right: 2, down: 1, left: 0 };
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

//----------------------------------------------------------------------------//
// Interactive demo trial: the instructed button is active, a fixed outcome is
// forced (for teaching), and a Continue button appears once the belief updates.
//----------------------------------------------------------------------------//
function make_demo_trial(cfg) {
    return {
        type: jsPsychHtmlKeyboardResponse,
        choices: "NO_KEYS",
        stimulus: `
            <div class="task-row">
                ${initialize_agent()}
                ${buttonStackHTML()}
                ${beliefPanelHTML()}
            </div>
            <div class="prompt">
                <div id="demo-instruction" class="demo-text">${cfg.instruction}</div>
                <button id="demo-continue" class="demo-continue" style="display:none;">Continue</button>
            </div>`,
        data: { task: "demo" },
        on_start: function () {
            agent_topPos = topPos0;
            agent_leftPos = leftPos0;
            if (cfg.reset) {
                for (const b of BUTTONS) {
                    for (const o of OUTCOMES) counts[b][o] = 0;
                }
            }
        },
        on_load: function () {
            refreshBeliefs();

            const active = document.getElementById("btn-" + cfg.button);
            const other = document.getElementById("btn-" + (cfg.button === "red" ? "blue" : "red"));
            // the other button looks normal but isn't clickable; it disappears on
            // selection, exactly as the unselected button does in the real task
            other.classList.add("inert");
            const cont = document.getElementById("demo-continue");

            active.addEventListener("click", function () {
                active.classList.add("disabled");
                other.classList.add("hidden");
                const outcome = cfg.outcome; // forced outcome for a predictable demo
                counts[cfg.button][outcome] += 1;
                moveAgent(outcome);
                setTimeout(function () {
                    refreshBeliefs(); // reveal the updated heatmap once the agent arrives
                    document.getElementById("demo-instruction").innerHTML = cfg.resultText;
                    cont.style.display = "inline-block";
                }, MOVE_MS);
            });

            cont.addEventListener("click", function () {
                jsPsych.finishTrial({ task: "demo", demo_button: cfg.button, demo_outcome: cfg.outcome });
            });
        }
    };
}

//----------------------------------------------------------------------------//
// Auto-play demo: the computer presses one button `cfg.outcomes.length` times,
// showing the agent move to each (fixed) outcome and return to the centre. Used
// to illustrate how reliable vs variable a button can be. No heatmap is shown
// (they haven't been introduced yet); the focus is on the movement. The Next
// button appears after `cfg.revealAfter` observations (default 5) so the
// participant may continue early while the rest keep playing.
//----------------------------------------------------------------------------//
function make_auto_demo_trial(cfg) {
    const revealAfter = cfg.revealAfter || 5;
    return {
        type: jsPsychHtmlKeyboardResponse,
        choices: "NO_KEYS",
        stimulus: `
            <div class="task-row">
                ${initialize_agent()}
                ${buttonStackHTML()}
            </div>
            <div class="prompt">
                <div id="demo-instruction" class="demo-text">${cfg.instruction}</div>
                <button id="demo-continue" class="demo-continue" style="display:none;">Next</button>
            </div>`,
        data: { task: "auto_demo", demo_button: cfg.button },
        on_start: function () {
            agent_topPos = topPos0;
            agent_leftPos = leftPos0;
        },
        on_load: function () {
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
                    moveAgent(seq[i]);
                    i += 1;
                    // let the participant continue once they've seen enough
                    if (i >= revealAfter) cont.style.display = "inline-block";
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
//   default    -> illustrative beliefs (blue uncertain between up & right, red
//                 strongly left), coin at the top; the button goes to its favoured
//                 outcome deterministically.
//   opts.useCurrentBeliefs -> keep the beliefs already learned (used by the
//                 early-stop practice); coin at a random location; outcome sampled
//                 from the true transition function.
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
                ${initialize_agent_gold()}
                ${buttonStackHTML()}
                ${beliefPanelHTML()}
            </div>
            <div class="prompt">
                <div id="demo-instruction" class="demo-text">${introText}</div>
                <h2 id="gold-feedback"></h2>
                <button id="demo-continue" class="demo-continue" style="display:none;">Continue</button>
            </div>`,
        data: { task: useCurrent ? "practice_gold" : "demo_gold" },
        on_start: function () {
            agent_topPos = topPos0;
            agent_leftPos = leftPos0;
            if (!useCurrent) {
                // illustrative beliefs: blue looks about as likely to go up as right
                // (i.e. uncertain), red strongly left.
                for (const b of BUTTONS) {
                    for (const o of OUTCOMES) counts[b][o] = 0;
                }
                counts.blue.up = 3;
                counts.blue.right = 3;
                counts.red.left = 3;
            }
        },
        on_load: function () {
            const goldOutcome = (opts.goldOutcome === "random")
                ? OUTCOMES[Math.floor(Math.random() * OUTCOMES.length)]
                : (opts.goldOutcome || "up");
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

                if (!SHOW_GOLD_OUTCOME) {
                    instr.innerHTML = `You chose the <strong>${button}</strong> button.<br>` +
                        `In this experiment the outcome is <strong>not revealed</strong> —<br>` +
                        `you won't find out whether you reached the coin,<br>` +
                        `so pick the button you most believe in.`;
                    cont.style.display = "inline-block";
                    return;
                }

                // reveal: sample the real outcome (practice) or use the favoured one (demo)
                const outcome = useCurrent ? sampleCategorical(TRUE_T[button]) : argmaxOutcome(button);
                const success = outcome === goldOutcome;
                moveAgent(outcome);
                setTimeout(function () {
                    const fb = document.getElementById("gold-feedback");
                    if (success) { fb.textContent = "You reached the coin!"; fb.style.color = "green"; }
                    else { fb.textContent = "You missed the coin."; fb.style.color = "red"; }
                    instr.innerHTML = opts.revealText ||
                        (`After selecting the final button,<br>` +
                         `you will find out whether you reached the coin.`);
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

    // ---- Overview + testing intro ----
    tl.push(instructionBlock([
        `<h2>How the room task works</h2>
         <p style="${P}">In the next phase you will explore a series of rooms</strong>, one at a time.</p>
         <p style="${P}">In each room you have <strong>${N_BUTTONS} coloured buttons</strong> to press, and there are
         <strong>${K_OUTCOMES} locations</strong> you can reach - i.e. up, down, left or right.</p>
         ${roomButtonsStaticHTML()}`,

        `<h2>Testing the buttons</h2>
         <p style="${P}">Each button takes you to one of these ${K_OUTCOMES} locations, but
         <strong>you don't know which location each button is most likely to lead to</strong>.</p>
         <p style="${P}">While there is <strong>always some degree of randomness</strong> in where a button leads,
         buttons can vary in how <strong>reliable</strong> they are.</p>
         <p style="${P}">Let's look at two examples.</p>
         ${roomButtonsStaticHTML()}`
    ]));

    // ---- Reliability animations (computer presses each button 10 times; Next after 5) ----
    tl.push(make_auto_demo_trial({
        button: "blue",
        outcomes: ["up", "up", "up", "left", "up", "up", "right", "up", "up", "up"],
        instruction: sayLines(
            `For example, a button may have a preferred direction, reliably leading you to one location.`,
            `See how the <strong>blue</strong> button often takes the agent <strong>upwards</strong> &mdash; although not all the time.`
        )
    }));
    tl.push(make_auto_demo_trial({
        button: "red",
        outcomes: ["up", "left", "right", "up", "down", "left", "down", "right", "up", "left"],
        instruction: sayLines(
            `Other buttons may be more random, taking you to many different locations.`,
            `See how the <strong>red</strong> button is much more variable in the outcomes it leads to.`
        )
    }));

    // ---- The heatmaps (introduced before the pressing/tick instructions) ----
    tl.push(instructionBlock([
        `<h2>The heatmaps</h2>
         <p style="${P}">To the right of the buttons you'll see a <strong>separate heatmap for each button</strong>,
         reflecting the history of outcomes you've observed for that button.</p>
         <p style="${P}">The two buttons are <strong>independent</strong>: each has its own heatmap, and the locations
         one button tends to reach <strong>may or may not overlap</strong> with those the other reaches.</p>
         <p style="${P}">Learning about one button therefore tells you nothing about the other.</p>
         <p style="${P}"><strong>So far, neither button below has been tried</strong>, so every location looks equally
         reachable — all four cells share the same faint colour. Only once you start testing do the colours diverge.</p>
         ${exampleDisplayStaticHTML(zeroCounts, zeroCounts)}`
    ]));

    // ---- Demos 1a / 1b: test each button once, watch its colour appear ----
    tl.push(make_demo_trial({
        button: "blue",
        outcome: "up",
        reset: true,
        instruction: sayLines(
            `Let's try it.`,
            `<strong>Click the blue button</strong> to test it.`
        ),
        resultText: sayLines(
            `The agent moved <strong>up</strong>.`,
            `The blue heatmap now shows stronger evidence that blue leads upward.`,
            `Click <strong>Continue</strong>.`
        )
    }));
    tl.push(make_demo_trial({
        button: "red",
        outcome: "left",
        reset: false,
        instruction: `Now <strong>click the red button</strong> to test the other one.`,
        resultText: sayLines(
            `The red button led <strong>left</strong>, so the red heatmap now shows stronger evidence for that outcome.`,
            `Notice this is <strong>independent</strong> of blue &mdash; each button has its own heatmap.`,
            `Click <strong>Continue</strong>.`
        )
    }));

    // ---- Demo 2: fading. Explanation stays on screen with the display. ----
    tl.push(make_demo_trial({
        button: "blue",
        outcome: "left",
        reset: false,
        instruction: sayLines(
            `When one location is reached, the colours for the other locations <strong>fade</strong>.`,
            `This is because the heatmap shows the <em>share</em> of observations that reached each location, so new evidence rebalances it.`,
            `<strong>Click the blue button again</strong> and watch.`
        ),
        resultText: sayLines(
            `The blue button led <strong>left</strong> this time.`,
            `Its "up" colour has <strong>faded</strong>, as the evidence now splits between up and left.`,
            `The two locations blue has <em>never</em> reached (right and down) have become <strong>even fainter</strong> &mdash; the more you sample without seeing them, the less reachable they look.`,
            `Click <strong>Continue</strong>.`
        )
    }));

    // ---- Testing the buttons: pressing, budget, and the tick to finish early.
    //      Full task display beneath. on_start sets up the practice room that
    //      follows (also runs on review, before the conditional practice trials). ----
    tl.push(instructionBlock([
        `<h2>Testing the buttons</h2>
         <p style="${P}">Each time you press a button, the agent moves and you see where it took you.</p>
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
    for (let t = 1; t <= N_TRIALS; t++) {
        tl.push({
            timeline: [make_trial(0, t, { practice: true })],
            conditional_function: function () { return !sampling_ended; }
        });
    }

    // ---- The gold-collection phase ----
    tl.push(instructionBlock([
        `<h2>Collecting the gold</h2>
         <p style="${P}">Once you've finished testing, a <strong>gold coin</strong> will appear at one of the
         ${K_OUTCOMES} locations.</p>
         <p style="${P}">You must then choose the <strong>button you think is most likely to take you to the
         coin</strong>, based on what you know about the buttons. Let's try a couple.</p>
         ${goldRoomStaticHTML("up")}`
    ]));

    // ---- Coin-selection demo 1: a reachable coin ----
    tl.push(make_gold_demo_trial());

    // ---- Coin-selection demo 2: a coin neither button reliably reaches ----
    tl.push(make_gold_demo_trial({
        goldOutcome: "down",
        instruction: sayLines(
            `Here's another coin.`,
            `Again, click the button you think is most likely to reach it.`
        ),
        revealText: sayLines(
            `This coin appeared in a location that <strong>neither button</strong> is likely to reach.`,
            `There's no guarantee the coin will appear somewhere a button can reliably reach &mdash; sometimes you simply won't be able to get it.`
        )
    }));

    // ---- End: start, or review the instructions again ----
    tl.push(make_review_choice_trial());

    return tl;
}
