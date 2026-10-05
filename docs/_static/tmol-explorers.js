/* Interactive explanations and an exact lookup of precomputed TMol scores. */
(() => {
  "use strict";
  const assetRoot = new URL(".", document.currentScript.src);
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  function selectButton(root, selected) {
    $$("button[aria-pressed]", root).forEach(button => {
      button.setAttribute("aria-pressed", String(button === selected));
    });
  }

  function initConcepts() {
    const circuit = $("#score-circuit");
    const explanations = {
      coordinates: "Atom positions enter every scoring term. A rendered scorer can be reused when only coordinates change; a new chemical or batch layout needs a new scorer.",
      contacts: "Atomic contacts include attraction, short-range repulsion, and electrostatics. Moving a side chain can relieve a clash while losing a favorable contact.",
      hydrogen: "Hydrogen-bond geometry and solvation contribute separate terms. A favorable contact depends on its surroundings as well as distance.",
      geometry: "Bond geometry and torsion preferences constrain the search. Backbone and side-chain preferences help distinguish otherwise similar contacts.",
      total: "Each contribution is multiplied by its score-function weight, then summed. The total has one value per pose. Inspect weighted terms with sum_terms=False.",
      gradient: "Autograd differentiates the total with respect to coordinates. An optimizer uses these derivatives to propose a move, evaluates it, and repeats. Packing instead searches discrete conformations."
    };
    if (circuit) $$("button", circuit).forEach(button => {
      button.addEventListener("click", () => {
        selectButton(circuit, button);
        $(".explorer-explanation", circuit).textContent = explanations[button.dataset.part];
      });
    });
    const forest = $("#fold-forest");
    const moves = {
      backbone: {nodes: ["A2", "A3"], text: "A backbone torsion along A1 → A2 moves downstream A2 and A3. Chain B is attached through A1, so this move leaves it fixed."},
      jump: {nodes: ["B1", "B2"], text: "The A1 → B1 jump moves B1 and B2 as a rigid body relative to chain A. Their internal coordinates stay unchanged."},
      other: {nodes: ["C2"], text: "A torsion in pose 1 moves its downstream C2 segment. Neither chain in pose 0 is affected: batch members have independent trees."}
    };
    if (forest) $$("button", forest).forEach(button => {
      button.addEventListener("click", () => {
        const move = moves[button.dataset.move];
        selectButton(forest, button);
        $$("[data-node]", forest).forEach(node => node.classList.toggle("active", move.nodes.includes(node.dataset.node)));
        $$("[data-edge]", forest).forEach(edge => edge.classList.toggle("active", edge.dataset.edge === button.dataset.move));
        $(".explorer-explanation", forest).textContent = move.text;
      });
    });
    const relax = $("#relax-schedule");
    const schedule = [[4.0, 5.1], [26.5, 28.0], [55.9, 58.1], [100, 100]];
    if (relax) $$("button", relax).forEach(button => {
      button.addEventListener("click", () => {
        const stage = Number(button.dataset.stage);
        const [pack, min] = schedule[stage];
        selectButton(relax, button);
        $("#relax-pack").textContent = `${pack.toFixed(1)}%`;
        $("#relax-min").textContent = `${min.toFixed(1)}%`;
        $(".explorer-explanation", relax).textContent = `Stage ${stage + 1} uses ${pack.toFixed(1)}% of the original repulsion weight for packing and ${min.toFixed(1)}% for minimization. The percentages describe weights, not measured energies.`;
      });
    });
  }

  function loadViewer() {
    if (window.$3Dmol) return Promise.resolve(window.$3Dmol);
    return new Promise((resolve, reject) => {
      const script = document.createElement("script");
      const timer = setTimeout(() => reject(new Error("Viewer download timed out")), 15000);
      script.src = new URL("vendor/3Dmol-2.4.2.min.js", assetRoot).href;
      script.onload = () => {
        clearTimeout(timer);
        window.$3Dmol ? resolve(window.$3Dmol) : reject(new Error("Viewer unavailable"));
      };
      script.onerror = () => { clearTimeout(timer); reject(new Error("Viewer download failed")); };
      document.head.appendChild(script);
    });
  }

  async function initPlayground(root) {
    $(".lab-loading", root).hidden = false;
    const updatePosters = () => {
      const mode = document.documentElement.dataset.theme === "dark" ? "dark" : "light";
      $(".lab-poster img", root).src = new URL(`playground/poster-${mode}.png`, assetRoot).href;
      $(".lab-scene-poster", root).src = new URL(`playground/protein-${mode}.png`, assetRoot).href;
    };
    updatePosters();
    new MutationObserver(updatePosters).observe(document.documentElement, {attributes: true, attributeFilter: ["data-theme"]});
    const response = await fetch(new URL("playground/phenylalanine.json", assetRoot), {signal: AbortSignal.timeout(15000)});
    if (!response.ok) throw new Error(`Dataset request failed (${response.status})`);
    const data = await response.json();
    if (data.schema_version !== 1 || data.frames.length !== data.angles.length ** 2) throw new Error("Unsupported dataset");
    const n = data.angles.length;
    const sliders = [$("#lab-chi1"), $("#lab-chi2")];
    sliders.forEach(input => { input.max = n - 1; });
    let current = data.start;
    let best = data.frames[current].total;
    let viewer;
    let protein;
    let movingModel;
    let pocketResidues = [];
    let renderRequest;
    let chartVisible = false;
    const baseline = data.frames[data.start];
    const partnerIndices = data.residue_labels.map((_, i) => i)
      .filter(i => i !== data.residue - 1)
      .sort((a, b) => Math.abs(baseline.partners[b] - data.frames[data.best].partners[b]) - Math.abs(baseline.partners[a] - data.frames[data.best].partners[a]))
      .slice(0, 8);
    let selectedPartner = partnerIndices[0];
    const partnerButtons = partnerIndices.map(index => {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.partner = index;
      button.innerHTML = "<span></span><strong></strong><small></small>";
      button.children[0].textContent = data.residue_labels[index];
      button.addEventListener("click", () => { selectedPartner = index; update(current); });
      $("#lab-partners").append(button);
      return button;
    });
    const chart = $("#lab-map");
    const context = chart.getContext("2d");
    const groupDefinitions = [
      ["Steric repulsion", ["fa_ljrep"]],
      ["Attractive contacts", ["fa_ljatr"]],
      ["Solvation", ["fa_lk", "lk_ball_iso", "lk_ball", "lk_bridge", "lk_bridge_uncpl"]],
      ["Electrostatics", ["fa_elec"]],
      ["Hydrogen bonds", ["hbond"]],
      ["Torsion preferences", ["rama", "omega", "dunbrack_rot", "dunbrack_rotdev", "dunbrack_semirot", "gen_torsions", "na_torsion", "na_torsion_well"]],
      ["Bond geometry", ["cart_lengths", "cart_angles", "cart_torsions", "cart_impropers", "cart_hxltorsions"]]
    ];
    const covered = new Set(groupDefinitions.flatMap(group => group[1]));
    groupDefinitions.push(["Other terms", data.terms.filter(term => !covered.has(term))]);
    const groups = groupDefinitions.map(([label, terms]) => ({label, terms, indices: terms.map(term => data.terms.indexOf(term)).filter(i => i >= 0)}));
    const tbody = $("#lab-terms");
    groups.forEach(group => {
      const row = document.createElement("tr");
      const heading = document.createElement("th");
      heading.scope = "row";
      heading.textContent = group.label;
      heading.title = group.terms.join(", ");
      row.append(heading, document.createElement("td"), document.createElement("td"));
      tbody.append(row);
      group.row = row;
    });
    const fmt = number => number.toLocaleString("en-US", {minimumFractionDigits: 2, maximumFractionDigits: 2});
    const signed = number => `${number > 0.005 ? "+" : ""}${fmt(Math.abs(number) < 0.005 ? 0 : number)}`;
    const changeClass = number => number < -0.005 ? "score-better" : number > 0.005 ? "score-worse" : "";
    const totals = data.frames.map(frame => frame.total);
    const minimum = Math.min(...totals);
    const range = Math.log1p(Math.max(...totals) - minimum);

    function drawChart() {
      if (!context || !chartVisible) return;
      const size = chart.width / n;
      data.frames.forEach((frame, i) => {
        const value = Math.log1p(frame.total - minimum) / range;
        context.fillStyle = `hsl(22 72% ${22 + value * 68}%)`;
        context.fillRect(Math.floor(i / n) * size, (n - 1 - i % n) * size, size, size);
      });
      const x = (Math.floor(current / n) + 0.5) * size;
      const y = (n - 1 - current % n + 0.5) * size;
      context.beginPath(); context.arc(x, y, size * 0.43, 0, Math.PI * 2);
      context.lineWidth = 3; context.strokeStyle = "#fff"; context.stroke();
      context.lineWidth = 1; context.strokeStyle = "#263238"; context.stroke();
    }

    function currentPdb() {
      return data.residue_pdb.map((line, i) => {
        const coords = data.frames[current].coordinates[i].map(v => v.toFixed(3).padStart(8, " ")).join("");
        return line.slice(0, 30) + coords + line.slice(54);
      }).join("\n") + "\nEND\n";
    }

    function drawProtein() {
      if (!viewer) return;
      if (movingModel) viewer.removeModel(movingModel);
      movingModel = viewer.addModel(currentPdb(), "pdb");
      movingModel.setStyle({}, {stick: {radius: 0.22, colorscheme: "orangeCarbon"}, sphere: {scale: 0.22, colorscheme: "orangeCarbon"}});
      protein.setStyle({}, {cartoon: {color: "#90a4ae", opacity: 0.55}});
      viewer.addStyle({model: 0, resi: pocketResidues}, {stick: {radius: 0.12, colorscheme: "grayCarbon"}});
      viewer.addStyle({model: 0, resi: selectedPartner + 1}, {stick: {radius: 0.2, colorscheme: "blueCarbon"}});
      viewer.render();
    }

    function update(index, message = "") {
      current = index;
      const frame = data.frames[index];
      best = Math.min(best, frame.total);
      sliders[0].value = Math.floor(index / n);
      sliders[1].value = index % n;
      sliders.forEach((slider, i) => {
        const angle = `${frame.angles[i]}°`;
        $(`#lab-chi${i + 1}-value`).textContent = angle;
        slider.setAttribute("aria-valuetext", `${frame.angles[i]} degrees relative to input`);
      });
      $("#lab-total").textContent = fmt(frame.total);
      $("#lab-total").dataset.value = frame.total;
      const delta = frame.total - baseline.total;
      $("#lab-change").textContent = signed(delta);
      $("#lab-change").className = changeClass(delta);
      $("#lab-best").textContent = fmt(best);
      groups.forEach(group => {
        const value = group.indices.reduce((sum, i) => sum + frame.terms[i], 0);
        const difference = group.indices.reduce((sum, i) => sum + frame.terms[i] - baseline.terms[i], 0);
        group.row.cells[1].textContent = fmt(value);
        group.row.cells[2].textContent = signed(difference);
        group.row.cells[2].className = changeClass(difference);
      });
      partnerButtons.forEach(button => {
        const index = Number(button.dataset.partner);
        const difference = frame.partners[index] - baseline.partners[index];
        button.setAttribute("aria-pressed", String(index === selectedPartner));
        button.children[1].textContent = fmt(frame.partners[index]);
        button.children[2].textContent = `Δ ${signed(difference)}`;
        button.children[2].className = changeClass(difference);
      });
      $("#lab-partner-detail").textContent = `PHE45 ↔ ${data.residue_labels[selectedPartner]}: ${fmt(frame.partners[selectedPartner])}, change ${signed(frame.partners[selectedPartner] - baseline.partners[selectedPartner])} from the start.`;
      $("#lab-feedback").textContent = message || (index === data.best ? "You found the lowest score on this grid." : delta < -0.005 ? "Improved. Which contributions account for the change?" : delta > 0.005 ? "The total increased. Compare the competing contributions." : "Try moving χ1 or χ2 to lower the score.");
      root.dataset.frame = index;
      const url = new URL(window.location.href);
      url.searchParams.set("chi1", frame.angles[0]);
      url.searchParams.set("chi2", frame.angles[1]);
      history.replaceState(null, "", url);
      drawChart();
      cancelAnimationFrame(renderRequest);
      renderRequest = requestAnimationFrame(drawProtein);
    }

    sliders.forEach(input => input.addEventListener("input", () => update(Number(sliders[0].value) * n + Number(sliders[1].value))));
    $("#lab-reset").addEventListener("click", () => { best = baseline.total; update(data.start); });
    $("#lab-reference").addEventListener("click", () => update(data.reference, "Input geometry restored. Both rotations are zero relative to the prepared input."));
    $("#lab-optimum").addEventListener("click", () => update(data.best, "Lowest score among the 576 sampled conformations. Other atoms were held fixed."));
    $(".lab-landscape").addEventListener("toggle", event => { chartVisible = event.target.open; drawChart(); });
    const moveOnChart = event => {
      const box = chart.getBoundingClientRect();
      const a = Math.max(0, Math.min(n - 1, Math.floor((event.clientX - box.left) / box.width * n)));
      const b = Math.max(0, Math.min(n - 1, n - 1 - Math.floor((event.clientY - box.top) / box.height * n)));
      update(a * n + b);
    };
    chart.addEventListener("pointerdown", event => { chart.setPointerCapture(event.pointerId); moveOnChart(event); });
    chart.addEventListener("pointermove", event => { if (chart.hasPointerCapture(event.pointerId)) moveOnChart(event); });
    chart.addEventListener("pointerup", event => { if (chart.hasPointerCapture(event.pointerId)) chart.releasePointerCapture(event.pointerId); });
    const incoming = new URL(window.location.href).searchParams;
    if (incoming.has("chi1") && incoming.has("chi2")) {
      const a = data.angles.indexOf(Number(incoming.get("chi1")));
      const b = data.angles.indexOf(Number(incoming.get("chi2")));
      if (a >= 0 && b >= 0) current = a * n + b;
    }
    $(".lab-content", root).hidden = false;
    $(".lab-poster", root).hidden = true;
    $(".lab-loading", root).hidden = true;
    update(current);

    try {
      const probe = document.createElement("canvas");
      const gl = probe.getContext("webgl") || probe.getContext("experimental-webgl");
      if (!gl) throw new Error("WebGL unavailable");
      gl.getExtension("WEBGL_lose_context")?.loseContext();
      const mol = await loadViewer();
      viewer = mol.createViewer($("#protein-viewer"), {antialias: true});
      protein = viewer.addModel(data.pdb, "pdb");
      const sidechain = protein.selectedAtoms({resi: data.residue});
      pocketResidues = [...new Set(protein.selectedAtoms({}).filter(atom => atom.resi !== data.residue && sidechain.some(other => (atom.x - other.x) ** 2 + (atom.y - other.y) ** 2 + (atom.z - other.z) ** 2 < 25)).map(atom => atom.resi))];
      const focus = () => { viewer.zoomTo({resi: data.residue}); viewer.zoom(0.62); viewer.render(); };
      $("#lab-focus").addEventListener("click", focus);
      $("#lab-whole").addEventListener("click", () => { viewer.zoomTo(); viewer.render(); });
      const theme = () => {
        viewer.setBackgroundColor(getComputedStyle(document.documentElement).getPropertyValue("--pst-color-surface").trim() || "#f5f7f8");
        viewer.render();
      };
      drawProtein(); theme(); focus();
      viewer.rotate(55, "y"); viewer.rotate(25, "x"); viewer.render();
      new MutationObserver(theme).observe(document.documentElement, {attributes: true, attributeFilter: ["data-theme"]});
      new ResizeObserver(() => viewer.resize()).observe($("#protein-viewer"));
      root.dataset.viewer = "ready";
    } catch (error) {
      root.dataset.viewerError = error.message;
      viewer = null;
      $("#protein-viewer").hidden = true;
      $(".lab-scene-poster", root).hidden = false;
      $("#viewer-status").textContent = "Static input geometry shown. 3D viewing is unavailable; the rotation controls, score table, and landscape still work. Try a browser with WebGL enabled.";
      $("#lab-focus").disabled = true;
      $("#lab-whole").disabled = true;
      root.dataset.viewer = "unavailable";
    }
  }

  function init() {
    initConcepts();
    const playground = $("#score-playground");
    if (playground) initPlayground(playground).catch(() => {
      $(".lab-loading", playground).textContent = "The scored dataset could not be loaded. Reload to try again, or download the dataset below.";
    });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init, {once: true});
  else init();
})();
