/* Browser checks against the built site and its exact, recorded scoring data.
 * NODE_PATH must contain Playwright. Pass the Sphinx HTML directory as argv[2].
 * Set TMOL_BROWSER_EXECUTABLE only when using an existing local Chromium.
 */
const { chromium } = require("playwright");
const fs = require("node:fs");
const path = require("node:path");
const http = require("node:http");
const assert = require("node:assert/strict");

const html = path.resolve(process.argv[2] || "docs/_build/html");
const data = JSON.parse(fs.readFileSync(path.join(html, "_static/playground/phenylalanine.json")));
const mime = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".png": "image/png", ".svg": "image/svg+xml" };
const server = http.createServer((request, response) => {
  const file = path.resolve(html, "." + decodeURIComponent(new URL(request.url, "http://localhost").pathname));
  if (!file.startsWith(html + path.sep) || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
    response.writeHead(404).end();
    return;
  }
  response.setHeader("Content-Type", mime[path.extname(file)] || "application/octet-stream");
  fs.createReadStream(file).pipe(response);
});

(async () => {
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.TMOL_BROWSER_EXECUTABLE || undefined,
    args: ["--enable-unsafe-swiftshader"],
  });
  const ready = page => page.waitForSelector('#score-playground[data-viewer="ready"]');
  const total = async page => Number(await page.locator("#lab-total").getAttribute("data-value"));
  try {
    for (const theme of ["light", "dark"]) for (const width of [1440, 390]) {
      const context = await browser.newContext({ viewport: { width, height: 1000 }, colorScheme: theme, reducedMotion: "reduce" });
      await context.addInitScript(theme => { localStorage.setItem("mode", theme); localStorage.setItem("theme", theme); }, theme);
      const page = await context.newPage();
      const errors = [];
      page.on("pageerror", error => errors.push(error.message));
      page.on("response", response => { if (response.status() >= 400) errors.push(`${response.status()} ${response.url()}`); });
      await page.route("**/*", route => {
        if (route.request().url().startsWith(base)) return route.continue();
        errors.push(`External dependency: ${route.request().url()}`);
        return route.abort();
      });
      await page.goto(`${base}/playground.html`);
      await ready(page);
      assert.equal(await total(page), data.frames[data.start].total);
      await page.locator("#lab-chi1").focus();
      await page.keyboard.press("ArrowRight");
      const a = Number(await page.locator("#lab-chi1").inputValue());
      const b = Number(await page.locator("#lab-chi2").inputValue());
      assert.equal(await total(page), data.frames[a * data.angles.length + b].total);
      await page.reload();
      await ready(page);
      assert.equal(await total(page), data.frames[a * data.angles.length + b].total, "URL restores the sampled state");
      await page.locator("#lab-optimum").click();
      assert.equal(await total(page), data.frames[data.best].total);
      const terms = await page.locator("#lab-terms tr td:nth-child(2)").allTextContents();
      assert(Math.abs(terms.reduce((sum, value) => sum + Number(value.replaceAll(",", "")), 0) - data.frames[data.best].total) < 0.05);
      const partner = page.locator("#lab-partners button").nth(1);
      await partner.click();
      assert.equal(await partner.getAttribute("aria-pressed"), "true");
      const partnerIndex = Number(await partner.getAttribute("data-partner"));
      const value = Number((await partner.locator("strong").innerText()).replaceAll(",", ""));
      assert(Math.abs(value - data.frames[data.best].partners[partnerIndex]) < 0.005);
      await page.locator("#lab-reference").click();
      assert.equal(await page.locator("#lab-chi1").inputValue(), "12");
      await page.locator(".lab-landscape summary").click();
      await page.locator("#lab-map").scrollIntoViewIfNeeded();
      const box = await page.locator("#lab-map").boundingBox();
      await page.mouse.move(box.x + 5, box.y + 5);
      await page.mouse.down();
      await page.mouse.move(box.x + box.width - 5, box.y + box.height - 5, { steps: 5 });
      await page.mouse.up();
      assert.equal(Number(await page.locator("#score-playground").getAttribute("data-frame")), (data.angles.length - 1) * data.angles.length);
      await page.locator(".lab-landscape summary").click();
      await page.locator("#lab-reset").click();
      assert.equal(await total(page), data.frames[data.start].total);
      assert(!(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)), `playground overflow: ${theme} ${width}`);
      await page.goto(`${base}/glossary.html`);
      await page.getByRole("button", { name: "Rigid-body jump", exact: true }).click();
      assert.deepEqual(await page.locator("#fold-forest .forest-node.active").evaluateAll(nodes => nodes.map(node => node.dataset.node)), ["B1", "B2"]);
      await page.getByRole("button", { name: "Stage 4", exact: true }).click();
      assert.equal(await page.locator("#relax-pack").innerText(), "100.0%");
      await page.locator('[data-part="gradient"]').click();
      assert((await page.locator("#score-circuit .explorer-explanation").innerText()).includes("Autograd"));
      assert(!(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)), `glossary overflow: ${theme} ${width}`);
      assert.deepEqual(errors, []);
      await context.close();
      console.log(`PASS ${theme}, ${width}px: native viewer, exact scores, URL state, keyboard, dragging, partners, glossary, local assets`);
    }
    for (const failure of ["viewer-download", "webgl"]) {
      const page = await browser.newPage();
      if (failure === "viewer-download") await page.route("**/3Dmol-2.4.2.min.js", route => route.abort());
      else await page.addInitScript(() => {
        const original = HTMLCanvasElement.prototype.getContext;
        HTMLCanvasElement.prototype.getContext = function (kind, ...args) {
          return kind.includes("webgl") ? null : original.call(this, kind, ...args);
        };
      });
      await page.goto(`${base}/playground.html`);
      await page.waitForSelector('[data-viewer="unavailable"]');
      assert(await page.locator(".lab-scene-poster").isVisible());
      await page.locator("#lab-optimum").click();
      assert.equal(await total(page), data.frames[data.best].total);
      await page.close();
      console.log(`PASS ${failure}: static structure and working score controls`);
    }
    const failedData = await browser.newPage();
    await failedData.route("**/phenylalanine.json", route => route.abort());
    await failedData.goto(`${base}/playground.html`);
    await failedData.getByText("The scored dataset could not be loaded.", { exact: false }).waitFor();
    assert(await failedData.locator(".lab-poster").isVisible());
    await failedData.close();
    const invalid = await browser.newPage();
    await invalid.goto(`${base}/playground.html?chi1=invalid&chi2=999`);
    await ready(invalid);
    assert.equal(await total(invalid), data.frames[data.start].total);
    await invalid.close();
    const nojs = await browser.newPage({ javaScriptEnabled: false });
    await nojs.goto(`${base}/playground.html`);
    assert(await nojs.locator("#score-playground noscript").isVisible());
    assert(await nojs.locator(".lab-poster img").evaluate(image => image.complete && image.naturalWidth > 0));
    await nojs.close();
    console.log("PASS unavailable data, invalid URL, and no-JavaScript poster");
  } finally {
    await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
})().catch(error => { console.error(error); server.close(); process.exitCode = 1; });
