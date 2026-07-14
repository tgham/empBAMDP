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

//----------------------------------------------------------------------------//
// Main room scaffold: base tile + centred agent (mirrors Experiment3 script.js)
//----------------------------------------------------------------------------//
function initialize_agent() {
    return `
    <div class="container">
        <img src="img/BaseAction.png" alt="Base Action Image" class="base-image">
        <div id="belief-overlay" class="belief-overlay"></div>
        <img id="agent" src="img/Agent.png" alt="Agent Image" class="agent-image">
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
        <img src="img/BaseAction.png" alt="Base Action Image" class="base-image">
        <div id="belief-overlay" class="belief-overlay"></div>
        <img id="gold" src="img/Goal.png" alt="Gold Coin" class="gold-image">
        <img id="agent" src="img/Agent.png" alt="Agent Image" class="agent-image">
    </div>`;
}

// Position the gold coin at a cardinal outcome cell.
function placeGold(outcome) {
    const { top, left } = outcomePercent(outcome);
    const gold = document.getElementById("gold");
    gold.style.top = top + "%";
    gold.style.left = left + "%";
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
function sampleTrueT() {
    for (const button of BUTTONS) {
        const probs = sampleDirichlet(ALPHA, K_OUTCOMES);
        TRUE_T[button] = {};
        OUTCOMES.forEach((outcome, i) => {
            TRUE_T[button][outcome] = probs[i];
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
    const cellPct = 100 / gridSize; // 33.33

    for (const outcome of OUTCOMES) {
        const idx = OUTCOME_CELL[outcome];
        const r = Math.floor(idx / gridSize);
        const col = idx % gridSize;

        const cell = document.createElement("div");
        cell.className = "overlay-cell";
        // inset within the cell so the base grid lines stay visible (same 0.1/0.8
        // convention as the target images in Experiment3)
        cell.style.left = `${col * cellPct + cellPct * 0.1}%`;
        cell.style.top = `${r * cellPct + cellPct * 0.1}%`;
        cell.style.width = `${cellPct * 0.8}%`;
        cell.style.height = `${cellPct * 0.8}%`;

        const pRed = posteriorMean("red", outcome);
        const pBlue = posteriorMean("blue", outcome);

        cell.innerHTML = `
            <svg viewBox="0 0 100 100" class="overlay-svg">
                <polygon points="0,0 100,0 100,100"
                         fill="rgb(${BTN_COLOR.red})" fill-opacity="${pRed}"></polygon>
                <polygon points="0,0 100,100 0,100"
                         fill="rgb(${BTN_COLOR.blue})" fill-opacity="${pBlue}"></polygon>
                <line x1="0" y1="0" x2="100" y2="100" stroke="#999" stroke-width="1.5"></line>
                <text x="72" y="30" class="overlay-num">${pRed.toFixed(2)}</text>
                <text x="28" y="76" class="overlay-num">${pBlue.toFixed(2)}</text>
            </svg>`;
        layer.appendChild(cell);
    }
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
