//----------------------------------------------------------------------------//
// Agent position state (percentage-based, reused from Experiment3 geometry)
//----------------------------------------------------------------------------//
const topPos0 = 50;
const leftPos0 = 50;
const gridSize = 3;
const step = 100 / gridSize; // 33.33% per cell
const MOVE_MS = 400; // agent move animation duration (must match #agent CSS transition)

let agent_topPos = topPos0;
let agent_leftPos = leftPos0;

// Fraction (1..0) of the gold coin remaining in this room's sampling phase.
// Depleted by 1/N_TRIALS on every button click (not on tick clicks).
let GOLD_FRACTION = 1;

function goldMaskStyle(fraction) {
    const f = Math.max(0, Math.min(1, fraction));
    if (f >= 1) return ""; // full coin: no mask needed, avoids the conic-gradient seam at the 360deg/0deg boundary
    if (f <= 0) return "-webkit-mask-image: none; mask-image: none; opacity: 0;"; // fully depleted: hide outright, same seam issue at 0deg
    const deg = f * 360;
    const mask = `conic-gradient(from 180deg, #000 ${deg}deg, transparent ${deg}deg)`;
    return `-webkit-mask-image:${mask}; mask-image:${mask}; -webkit-mask-size:100% 100%; mask-size:100% 100%;`;
}

// The depleting coin shown top-centre during sampling.
function goldCostCoinHTML() {
    return `
        <div class="gold-cost-wrap">
            <img id="gold-cost-coin" src="img/Goal.png" alt="Gold coin remaining"
                 class="gold-cost-coin" style="${goldMaskStyle(GOLD_FRACTION)}">
        </div>`;
}

// Redraw the top-centre coin from the current GOLD_FRACTION.
function updateGoldCostCoin() {
    const el = document.getElementById("gold-cost-coin");
    if (el) el.setAttribute("style", goldMaskStyle(GOLD_FRACTION));
}

//----------------------------------------------------------------------------//
// Screen scaffold, shared by every screen of the task and the instructions. See
// css/style.css for the geometry: the stage sits dead centre and the text is
// positioned off that centre, so the room lands in the same place on every
// screen and no amount of text can shift it.
//   title  optional heading
//   lines  the sentences above the stage -- one <p>, i.e. one line, each
//   stage  the room / buttons / illustration (omit for a text-only screen)
//   below  a note to sit just under the stage
// The sentences live in #screen-lines so a trial can rewrite them in place --
// see showScreenFeedback, which is how a result replaces the prompt.
//----------------------------------------------------------------------------//
function linesHTML(lines) {
    return (lines || []).map((line) => `<p>${line}</p>`).join("");
}

function screenHTML(opts) {
    const title = opts.title ? `<div class="screen-title">${opts.title}</div>` : ``;
    return `
        <div class="screen">
            <div class="screen-text">
                ${title}
                <div id="screen-lines">${linesHTML(opts.lines)}</div>
            </div>
            ${opts.gap ? `<div class="screen-gap">${opts.gap}</div>` : ``}
            ${opts.stage ? `<div class="screen-stage">${opts.stage}</div>` : ``}
            ${opts.below ? `<div class="screen-below">${opts.below}</div>` : ``}
        </div>`;
}

// Replace a screen's text with the outcome of the choice just made, so the
// result reads where the prompt was (above the room). `extraLines` are any
// further sentences to show under it.
function showScreenFeedback(message, ok, extraLines) {
    const el = document.getElementById("screen-lines");
    if (!el) return;
    el.innerHTML =
        `<div class="screen-feedback" style="color:${ok ? "#2ca02c" : "#c0392b"}">${message}</div>` +
        linesHTML(extraLines);
}

//----------------------------------------------------------------------------//
// Main room scaffold: base tile + centred agent (mirrors Experiment3 script.js)
//----------------------------------------------------------------------------//
function initialize_agent() {
    return `
    <div class="container">
        <img src="img/BaseAction_4k.png" alt="Base Action Image" class="base-image" decoding="sync" fetchpriority="high">
        <div id="belief-overlay" class="belief-overlay"></div>
        <img id="agent" src="img/Agent.png" alt="Agent Image" class="agent-image" decoding="sync">
    </div>`;
}

//----------------------------------------------------------------------------//
// Percentage top/left of a cardinal outcome cell (centre is topPos0/leftPos0).
//----------------------------------------------------------------------------//
function outcomePercent(outcome) {
    let top = topPos0;
    let left = leftPos0;
    if (outcome === "up") top -= step;
    if (outcome === "down") top += step;
    if (outcome === "left") left -= step;
    if (outcome === "right") left += step;
    return { top, left };
}

function goldInRoomStaticHTML(outcome) {
    const { top, left } = outcomePercent(outcome);
    return `<img src="img/Goal.png" alt="Gold Coin" class="gold-image" style="top:${top}%; left:${left}%;">`;
}

//----------------------------------------------------------------------------//
// Move the agent to a cardinal outcome cell. Same nudging logic as Experiment3.
//----------------------------------------------------------------------------//
function moveAgent(outcome) {
    const { top, left } = outcomePercent(outcome);
    agent_topPos = top;
    agent_leftPos = left;
    const agent = document.getElementById("agent");
    agent.style.top = top + "%";
    agent.style.left = left + "%";
}

//----------------------------------------------------------------------------//
// Gold-collection scaffold: base tile + belief overlay + a gold coin + agent.
//----------------------------------------------------------------------------//
function initialize_agent_gold() {
    return `
    <div class="container">
        <img src="img/BaseAction_4k.png" alt="Base Action Image" class="base-image" decoding="sync" fetchpriority="high">
        <div id="belief-overlay" class="belief-overlay"></div>
        <img id="gold" src="img/Goal.png" alt="Gold Coin" class="gold-image">
        <img id="agent" src="img/Agent.png" alt="Agent Image" class="agent-image" decoding="sync">
    </div>`;
}

// Position the gold coin at a cardinal outcome cell.
function placeGold(outcome) {
    const { top, left } = outcomePercent(outcome);
    const gold = document.getElementById("gold");
    gold.style.top = top + "%";
    gold.style.left = left + "%";
    gold.style.cssText += goldMaskStyle(GOLD_FRACTION);   // <-- new
}

//----------------------------------------------------------------------------//
// Sample a categorical outcome from a distribution over OUTCOMES.
//----------------------------------------------------------------------------//
function sampleCategorical(dist) {
    const r = Math.random();
    let cum = 0;
    for (const outcome of OUTCOMES) {
        cum += dist[outcome];
        if (r < cum) return outcome;
    }
    return OUTCOMES[OUTCOMES.length - 1]; // guard against float rounding
}

//----------------------------------------------------------------------------//
// Dirichlet sampling: draw each button's hidden transition distribution from a
// symmetric Dirichlet(ALPHA) prior (a Dirichlet sample = normalised independent
// Gamma(ALPHA) draws). Uses Marsaglia-Tsang, with the boosting trick for the
// ALPHA<1 regime we operate in.
//----------------------------------------------------------------------------//
function randn() {
    // standard normal via Box-Muller
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function sampleGamma(shape) {
    // Gamma(shape, scale=1)
    if (shape < 1) {
        return sampleGamma(shape + 1) * Math.pow(Math.random(), 1 / shape);
    }
    const d = shape - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    while (true) {
        let x, v;
        do {
            x = randn();
            v = 1 + c * x;
        } while (v <= 0);
        v = v * v * v;
        const u = Math.random();
        if (u < 1 - 0.0331 * x * x * x * x) return d * v;
        if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
}

function sampleDirichlet(alpha, k) {
    const g = [];
    for (let i = 0; i < k; i++) g.push(sampleGamma(alpha));
    const s = g.reduce((a, b) => a + b, 0);
    return g.map((x) => x / s);
}

// Draw a fresh hidden transition distribution for every button and store in TRUE_T.
// Under CONTEXTUAL each button's prior is chosen by an independent fair coin flip
// between ALPHA_CTX1 and ALPHA_CTX2; the draw is recorded in BUTTON_CTX.
function sampleTrueT() {
    for (const button of BUTTONS) {
        let alpha = ALPHA;
        if (CONTEXTUAL) {
            BUTTON_CTX[button] = Math.random() < 0.5 ? 1 : 2;
            alpha = BUTTON_CTX[button] === 1 ? ALPHA_CTX1 : ALPHA_CTX2;
        } else {
            BUTTON_CTX[button] = null;
        }
        const probs = sampleDirichlet(alpha, K_OUTCOMES);
        TRUE_T[button] = {};
        OUTCOMES.forEach((outcome, i) => {
            TRUE_T[button][outcome] = probs[i];
        });
    }
}

//----------------------------------------------------------------------------//
// Preset rooms: draw each button's hidden transition function from the Dirichlet
// POSTERIOR given the observations the participant is about to watch, i.e.
// Dirichlet(ALPHA + count_k) per button. This keeps the demonstrated outcomes
// consistent with the room's real dynamics, so the gold trial (and the bonus)
// actually reward reading the evidence. A button with no observations falls back
// to the plain Dirichlet(ALPHA) prior, which is exactly what the formula gives.
//
// Note this deliberately differs from gen_emp in the modelling code, which resets
// the environment and draws the true T independently of the seeded history.
//----------------------------------------------------------------------------//
function sampleTrueTFromPreset(presetCounts) {
    for (const button of BUTTONS) {
        const cnts = (presetCounts && presetCounts[button]) || freshOutcomeMap();
        const g = OUTCOMES.map((o) => sampleGamma(ALPHA + (cnts[o] || 0)));
        const s = g.reduce((a, b) => a + b, 0);
        BUTTON_CTX[button] = null;   // contexts are unused in the preset design
        TRUE_T[button] = {};
        OUTCOMES.forEach((outcome, i) => {
            TRUE_T[button][outcome] = g[i] / s;
        });
    }
}

//----------------------------------------------------------------------------//
// Dirichlet posterior mean that `button` reaches `outcome`.
//   (alpha + count_k) / (K*alpha + N_button)
//----------------------------------------------------------------------------//
function posteriorMean(button, outcome) {
    const total = OUTCOMES.reduce((acc, o) => acc + counts[button][o], 0);
    return (ALPHA + counts[button][outcome]) / (K_OUTCOMES * ALPHA + total);
}

//----------------------------------------------------------------------------//
// Snapshots for data logging (deep copies of the current belief state).
//----------------------------------------------------------------------------//
function countsSnapshot() {
    return JSON.parse(JSON.stringify(counts));
}
function posteriorSnapshot() {
    const snap = {};
    for (const button of BUTTONS) {
        snap[button] = {};
        for (const outcome of OUTCOMES) snap[button][outcome] = posteriorMean(button, outcome);
    }
    return snap;
}

//----------------------------------------------------------------------------//
// A colorbar (probability 0 bottom -> 1 top) in a button's colour. Composited
// over white to match the cell shading (rgba over white).
//----------------------------------------------------------------------------//
function colorbarHTML(button) {
    const c = BTN_COLOR[button];
    return `
        <div class="colorbar-wrap">
            <div class="colorbar-ticks"><span>1</span><span>0.5</span><span>0</span></div>
            <div class="colorbar" style="background: linear-gradient(to top, rgba(${c},0), rgba(${c},1)), #fff;"></div>
        </div>`;
}

//----------------------------------------------------------------------------//
// Belief-block titles double as sample counters, e.g. "Blue: 3 samples".
//----------------------------------------------------------------------------//
function sampleCount(button) {
    return OUTCOMES.reduce((acc, o) => acc + counts[button][o], 0);
}
function beliefLabelText(button) {
    const n = sampleCount(button);
    const name = button.charAt(0).toUpperCase() + button.slice(1);
    return `${name}: ${n} sample${n === 1 ? "" : "s"}`;
}
function updateBeliefLabels() {
    for (const button of BUTTONS) {
        const el = document.getElementById("label-" + button);
        if (el) el.textContent = beliefLabelText(button);
    }
}

// A labelled colorbar on its own (used as a legend in overlay mode).
function colorbarLegendHTML(button) {
    return `
        <div class="belief-block">
            <div class="belief-label" id="label-${button}">${beliefLabelText(button)}</div>
            ${colorbarHTML(button)}
        </div>`;
}

//----------------------------------------------------------------------------//
// HTML for one belief block: sample-counter title, the 3x3 grid, and its colorbar.
//----------------------------------------------------------------------------//
function beliefBlockHTML(button) {
    return `
        <div class="belief-block">
            <div class="belief-label" id="label-${button}">${beliefLabelText(button)}</div>
            <div class="belief-body">
                <div class="belief-grid" id="belief-${button}"></div>
                ${colorbarHTML(button)}
            </div>
        </div>`;
}

//----------------------------------------------------------------------------//
// Overlay mode: shade the four cardinal cells of the MAIN grid with BOTH buttons'
// posteriors, split diagonally -- red = upper-right triangle, blue = lower-left.
//----------------------------------------------------------------------------//
function renderMainBeliefOverlay() {
    const layer = document.getElementById("belief-overlay");
    if (!layer) return;
    layer.innerHTML = "";
    const cellPct = 100 / gridSize;

    for (const outcome of OUTCOMES) {
        const idx = OUTCOME_CELL[outcome];
        const r = Math.floor(idx / gridSize);
        const col = idx % gridSize;

        const cell = document.createElement("div");
        cell.className = "overlay-cell";
        cell.style.left = `${col * cellPct + cellPct * 0.1}%`;
        cell.style.top = `${r * cellPct + cellPct * 0.1}%`;
        cell.style.width = `${cellPct * 0.8}%`;
        cell.style.height = `${cellPct * 0.8}%`;

        const stripeW = 100 / BUTTONS.length;
        let svg = "";
        BUTTONS.forEach((button, i) => {
            const p = posteriorMean(button, outcome);
            const x = i * stripeW;
            svg += `<rect x="${x}" y="0" width="${stripeW}" height="100"
                        fill="rgb(${BTN_COLOR[button]})" fill-opacity="${p}"></rect>`;
            svg += `<text x="${x + stripeW / 2}" y="54" text-anchor="middle" class="overlay-num">${p.toFixed(2)}</text>`;
        });
        for (let i = 1; i < BUTTONS.length; i++) {
            const x = i * stripeW;
            svg += `<line x1="${x}" y1="0" x2="${x}" y2="100" stroke="#999" stroke-width="1.5"></line>`;
        }

        cell.innerHTML = `<svg viewBox="0 0 100 100" class="overlay-svg" preserveAspectRatio="none">${svg}</svg>`;
        layer.appendChild(cell);
    }
}

// `outcomeCounts`: {button: count} — how many times each button has led to
// this outcome so far. `buttonOrder`: [top, bottomLeft, bottomRight] — the
// room's triangle layout. Displayed left-to-right as
// [bottomLeft, top, bottomRight], so each token column lines up under/above
// its corresponding button.
function counterCellHTML(outcomeCounts, highlightButton, buttonOrder) {
    buttonOrder = buttonOrder || BUTTON_ORDER;
    if (!Array.isArray(buttonOrder) || buttonOrder.length !== BUTTONS.length) {
        console.error("counterCellHTML: invalid buttonOrder", buttonOrder);
        buttonOrder = BUTTONS.slice();
    }
    const rows = Math.ceil(N_TRIALS / 2);

    function section(button, n, mirrored) {
        let slots = "";
        for (let k = 0; k < N_TRIALS; k++) {
            let idx = k;
            if (mirrored) {
                const partner = (k % 2 === 0) ? k + 1 : k - 1;
                if (partner < N_TRIALS) idx = partner;
            }
            const isNew = button === highlightButton && idx === n - 1;
            slots += (idx < n)
                ? `<div class="counter-slot"><div class="counter-token${isNew ? " token-new" : ""}" style="background:rgb(${BTN_COLOR[button]})"></div></div>`
                : `<div class="counter-slot"></div>`;
        }
        return `<div class="counter-section" style="grid-template-rows:repeat(${rows},1fr)">${slots}</div>`;
    }

    // buttonOrder is [top, bottomLeft, bottomRight]; remap to display order
    // [bottomLeft, top, bottomRight] so each column matches its button's
    // horizontal position in the triangle.
    const [topButton, bottomLeftButton, bottomRightButton] = buttonOrder;
    const leftButton = bottomLeftButton;
    const midButton = topButton;
    const rightButton = bottomRightButton;

    return (
        section(leftButton, outcomeCounts[leftButton], false) +
        section(midButton, outcomeCounts[midButton], false) +
        section(rightButton, outcomeCounts[rightButton], true)
    );
}

// build the counter tokens into the given overlay layer from a counts object.
// `highlight` ({button, outcome}) optionally animates the just-placed token.
function renderCountersInto(layer, cnts, highlight, buttonOrder) {
    if (!layer) return;
    layer.classList.add("counter-layer");
    layer.innerHTML = "";
    const cellPct = 100 / gridSize;
    for (const outcome of OUTCOMES) {
        const idx = OUTCOME_CELL[outcome];
        const r = Math.floor(idx / gridSize);
        const col = idx % gridSize;
        const cell = document.createElement("div");
        cell.className = "counter-cell";
        cell.style.left = `${col * cellPct + cellPct * 0.08}%`;
        cell.style.top = `${r * cellPct + cellPct * 0.08}%`;
        cell.style.width = `${cellPct * 0.84}%`;
        cell.style.height = `${cellPct * 0.84}%`;
        const hl = highlight && highlight.outcome === outcome ? highlight.button : null;
        const outcomeCounts = {};
        for (const b of BUTTONS) outcomeCounts[b] = cnts[b][outcome];
        cell.innerHTML = counterCellHTML(outcomeCounts, hl, buttonOrder);
        layer.appendChild(cell);
    }
}

// live: render from the global counts into the main grid's overlay layer.
// pass {button, outcome} to pop in the token just added for that observation.
function renderMainCounters(highlight, buttonOrder) {
    renderCountersInto(document.getElementById("belief-overlay"), counts, highlight, buttonOrder);
}

// static: a container (base tile + counter tokens + agent) for instruction slides
function roomCountersStaticHTML(countsByButton, buttonOrder, goldOutcome) {
    const cellPct = 100 / gridSize;
    let cells = "";
    for (const outcome of OUTCOMES) {
        const idx = OUTCOME_CELL[outcome];
        const r = Math.floor(idx / gridSize);
        const col = idx % gridSize;
        const style = `left:${col * cellPct + cellPct * 0.08}%; top:${r * cellPct + cellPct * 0.08}%;` +
                      `width:${cellPct * 0.84}%; height:${cellPct * 0.84}%;`;
        const outcomeCounts = {};
        for (const b of BUTTONS) {
            outcomeCounts[b] = (countsByButton[b] && countsByButton[b][outcome]) || 0;
        }
        cells += `<div class="counter-cell" style="${style}">${counterCellHTML(outcomeCounts, null, buttonOrder)}</div>`;
    }
    return `
        <div class="container">
            <img src="img/BaseAction_4k.png" alt="Base" class="base-image" decoding="sync" fetchpriority="high">
            <div class="belief-overlay counter-layer">${cells}</div>
            ${goldOutcome ? goldInRoomStaticHTML(goldOutcome) : ""}
            <img src="img/Agent.png" alt="Agent" class="agent-image" decoding="sync">
        </div>`;
}

//----------------------------------------------------------------------------//
// Render a belief grid: 3x3 of cells, cardinals shaded by posterior mean in the
// button's colour, centre cell holds a small coloured dot, corners blank.
//----------------------------------------------------------------------------//
function renderBeliefGrid(button, containerEl) {
    containerEl.innerHTML = "";
    // map cell index -> outcome (or null)
    const cellOutcome = {};
    for (const outcome of OUTCOMES) cellOutcome[OUTCOME_CELL[outcome]] = outcome;

    for (let j = 0; j < gridSize * gridSize; j++) {
        const cell = document.createElement("div");
        cell.className = "belief-cell";

        if (j === 4) {
            // centre: identifying coloured dot
            const dot = document.createElement("div");
            dot.className = "belief-dot";
            dot.style.background = `rgb(${BTN_COLOR[button]})`;
            cell.appendChild(dot);
        } else if (cellOutcome[j]) {
            const outcome = cellOutcome[j];
            const p = posteriorMean(button, outcome);
            cell.style.background = `rgba(${BTN_COLOR[button]}, ${p})`;
            const label = document.createElement("span");
            label.className = "belief-num";
            label.textContent = p.toFixed(2);
            cell.appendChild(label);
        }
        // corners left blank

        containerEl.appendChild(cell);
    }
}
