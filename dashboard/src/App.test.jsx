import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { readFileSync } from "node:fs";
import { isAbsolute } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const fixturePath =
  process.env.DASHBOARD_CONTRACT_FIXTURE ?? `${process.cwd()}/src/generated/release.json`;
if (!fixturePath || !isAbsolute(fixturePath)) {
  throw new Error("DASHBOARD_CONTRACT_FIXTURE must name the absolute builder-emitted JSON fixture");
}
const releaseFixture = JSON.parse(readFileSync(fixturePath, "utf8"));
const styles = readFileSync(`${process.cwd()}/src/styles.css`, "utf8");
const studyViews = readFileSync(`${process.cwd()}/src/views/StudyViews.jsx`, "utf8");

vi.mock("virtual:dashboard-release", async () => {
  const { readFileSync: readFixture } = await import("node:fs");
  const path =
    process.env.DASHBOARD_CONTRACT_FIXTURE ?? `${process.cwd()}/src/generated/release.json`;
  return { default: JSON.parse(readFixture(path, "utf8")) };
});

import { App, ErrorBoundary } from "./main.jsx";
import { comparisonInterpretation } from "./views/EvidenceViews.jsx";

afterEach(() => {
  cleanup();
  document.documentElement.removeAttribute("data-theme");
  document.documentElement.removeAttribute("lang");
  vi.restoreAllMocks();
});

beforeEach(() => {
  const values = new Map();
  Object.defineProperty(window, "localStorage", {
    configurable: true,
    value: {
      getItem: (key) => values.get(key) ?? null,
      removeItem: (key) => values.delete(key),
      setItem: (key, value) => values.set(key, String(value)),
    },
  });
});

describe("builder-owned dashboard contract", () => {
  it("detects Turkish, persists an explicit language, and translates the shell", () => {
    vi.spyOn(window.navigator, "language", "get").mockReturnValue("tr-TR");
    render(<App release={releaseFixture} />);

    expect(document.documentElement.lang).toBe("tr");
    expect(screen.getByRole("heading", { name: "Günlük ve saatlik şebeke yükü tahminleri" })).toBeTruthy();

    const language = screen.getByRole("combobox", { name: "Dil" });
    fireEvent.change(language, { target: { value: "en" } });

    expect(document.documentElement.lang).toBe("en");
    expect(window.localStorage.getItem("dashboard-language")).toBe("en");
    expect(screen.getByRole("heading", { name: "Daily and hourly grid-load forecasts" })).toBeTruthy();
    expect(document.body.textContent).not.toMatch(/Final-period classification|Retrospective; previously inspected/);
  });

  it("packages the existing app below the HTTP-served docs entry with relative URLs", () => {
    const dashboardPackage = JSON.parse(readFileSync(`${process.cwd()}/package.json`, "utf8"));
    const viteConfig = readFileSync(`${process.cwd()}/vite.config.js`, "utf8");
    const docsEntry = readFileSync(`${process.cwd()}/../docs/index.html`, "utf8");

    expect(dashboardPackage.scripts["build:docs"]).toBe(
      "vite build --outDir ../docs/dashboard --emptyOutDir",
    );
    expect(viteConfig).toMatch(/base:\s*["']\.\/["']/);

    const entryDocument = new DOMParser().parseFromString(docsEntry, "text/html");
    const refresh = entryDocument.querySelector('meta[http-equiv="refresh" i]');
    const fallback = entryDocument.querySelector('a[href="./dashboard/index.html"]');
    expect(refresh?.content).toMatch(/url=\.\/dashboard\/index\.html$/i);
    expect(fallback?.textContent.trim()).toBeTruthy();
    for (const element of entryDocument.querySelectorAll("[src], [href]")) {
      const target = element.getAttribute("src") ?? element.getAttribute("href");
      expect(target).not.toMatch(/^\/|^[a-z][a-z\d+.-]*:/i);
    }
  });

  it("loads the generated fixture with separate point-error and interval horizons", () => {
    expect(releaseFixture.fixture_notice).toBeUndefined();
    expect(releaseFixture.selector_options.hourly_point_error_horizons).toEqual(
      Array.from({ length: 24 }, (_, index) => index + 1),
    );
    expect(releaseFixture.selector_options.hourly_interval_coverage_horizons).toEqual([24]);
    const pointModels = releaseFixture.selector_options.hourly_point_error_models;
    const pointHorizons = releaseFixture.selector_options.hourly_point_error_horizons;
    expect(releaseFixture.hourly_error.horizon_profile).toHaveLength(
      pointModels.length * pointHorizons.length,
    );
    for (const model of pointModels) {
      expect(releaseFixture.hourly_error.horizon_profile.filter((row) => row.model === model).map((row) => row.horizon)).toEqual(pointHorizons);
    }
    expect(new Set(releaseFixture.intervals.hourly.filter((row) => row.slice_type === "horizon").map((row) => Number(row.slice_value)))).toEqual(new Set([24]));
  });

  it("renders four ordered chapters with exactly 3, 3, 3, and 1 complete figures", () => {
    render(<App release={releaseFixture} />);
    const chapters = [...document.querySelectorAll(".study-chapter")];
    expect(chapters.map((chapter) => chapter.querySelector(":scope > h2").textContent)).toEqual([
      "Demand patterns",
      "Forecast performance",
      "Reliability and weak points",
      "Evaluation method",
    ]);
    expect(chapters.map((chapter) => chapter.querySelectorAll(":scope > .chapter-figures > .chapter-figure > [data-chart-frame]").length)).toEqual([3, 3, 3, 1]);
    expect([...screen.getByRole("navigation", { name: "Results sections" }).querySelectorAll("a")].map((link) => link.textContent)).toEqual([
      "Demand patterns",
      "Forecast performance",
      "Reliability and weak points",
      "Evaluation method",
    ]);
    expect(document.querySelectorAll("[data-summary]")).toHaveLength(3);
    for (const frame of document.querySelectorAll(".study-chapter [data-chart-frame]")) {
      expect(frame.querySelector(".figure-layout > .chart")).toBeTruthy();
      expect(frame.querySelector(".figure-layout > aside[data-interpretation]")).toBeTruthy();
      expect(frame.querySelector("figcaption").textContent).toMatch(/n =/i);
      expect(frame.querySelector("details table")).toBeTruthy();
    }
    expect(document.querySelector("#forecasts, #intervals, #hourly-error, #reconciliation, #comparison, #protocol")).toBeNull();
  });

  it("keeps point-error h1-h24 independent from h24-only interval coverage", () => {
    render(<App release={releaseFixture} />);
    const reliability = document.querySelector("#reliability");
    const errorFigure = reliability.querySelectorAll(".chapter-figure")[0];
    fireEvent.change(within(errorFigure).getByRole("combobox", { name: "Frequency" }), { target: { value: "hourly" } });
    const pointHorizon = within(errorFigure).getByRole("combobox", { name: "Forecast horizon" });
    expect([...pointHorizon.options].map((option) => option.value)).toEqual([
      "all",
      ...Array.from({ length: 24 }, (_, index) => `h${index + 1}`),
    ]);

    const intervalFigure = reliability.querySelectorAll(".chapter-figure")[1];
    fireEvent.change(within(intervalFigure).getByRole("combobox", { name: "Frequency" }), { target: { value: "hourly" } });
    fireEvent.change(within(intervalFigure).getByRole("combobox", { name: "Interval scope" }), { target: { value: "horizon" } });
    const intervalHorizon = within(intervalFigure).getByRole("combobox", { name: "Interval horizon" });
    expect([...intervalHorizon.options].map((option) => option.value)).toEqual(["24"]);
    expect(within(intervalFigure).getByText(/h24, final evaluation/i)).toBeTruthy();
  });

  it("uses generated chapter selectors and reader-facing evidence roles", () => {
    render(<App release={releaseFixture} />);
    const loadProfile = within(document.querySelector("#load-patterns")).getByRole("combobox", { name: "Demand grouping" });
    expect([...loadProfile.options].map((option) => option.value)).toEqual(["weekday", "month"]);

    const comparisonFigure = document.querySelector("#forecast-performance").querySelectorAll(".chapter-figure")[1];
    fireEvent.change(within(comparisonFigure).getByRole("combobox", { name: "Frequency" }), { target: { value: "hourly" } });
    expect([...within(comparisonFigure).getByRole("combobox", { name: "Evaluation period" }).options].map((option) => option.textContent)).toEqual([
      "Validation",
      "Final evaluation",
    ]);
    fireEvent.change(within(comparisonFigure).getByRole("combobox", { name: "Comparison view" }), { target: { value: "uncertainty" } });
    expect(within(comparisonFigure).getByLabelText("What this shows").textContent).toMatch(/practically tied.*range includes zero/i);
    expect(document.body.textContent).not.toMatch(/\btest\b|test_MAPE|ensemble_MAPE|reconciled_h24_hourly_MAE/);
  });

  it("places every model-comparison label on a categorical vertical axis", () => {
    expect(studyViews).toMatch(
      /title=\{pick\("Model and baseline comparison"[\s\S]*?<BarChart[^>]*layout="vertical"[\s\S]*?<YAxis[^>]*type="category"[^>]*dataKey="label"/,
    );
  });

  it("handles equality, constant slices, cross-zero uncertainty, and bounded many-row commentary", () => {
    const equalRelease = structuredClone(releaseFixture);
    equalRelease.study.performance.daily_comparison.forEach((row) => { row.MAE = 100; });
    equalRelease.hourly_error.horizon_profile.forEach((row) => { row.MAE = 180; });
    render(<App release={equalRelease} />);
    const comparisonFigure = document.querySelector("#forecast-performance").querySelectorAll(".chapter-figure")[1];
    expect(within(comparisonFigure).getByLabelText("What this shows").textContent).toMatch(/all 7 displayed models have equal MAE/i);
    const errorFigure = document.querySelector("#reliability").querySelectorAll(".chapter-figure")[0];
    fireEvent.change(within(errorFigure).getByRole("combobox", { name: "Frequency" }), { target: { value: "hourly" } });
    expect(within(errorFigure).getByLabelText("What this shows").textContent).toMatch(/error is the same across displayed horizons/i);

    const base = { candidate: "candidate", reference: "reference", mae_difference: 0, ci_lower: 0, ci_upper: 0, practical_tie: false };
    expect(comparisonInterpretation([base])).toMatch(/equal MAE/i);
    expect(comparisonInterpretation([{ ...base, mae_difference: 1, ci_lower: -2, ci_upper: 3 }])).toMatch(/no clear lower-error result/i);
    expect(comparisonInterpretation([{ ...base, mae_difference: 1, ci_lower: -2, ci_upper: 3, practical_tie: true }])).toMatch(/practically tied/i);
    const many = comparisonInterpretation([
      base,
      { ...base, candidate: "tie", mae_difference: 1, ci_lower: -2, ci_upper: 3, practical_tie: true },
      { ...base, candidate: "unclear", mae_difference: 1, ci_lower: -2, ci_upper: 3 },
      { ...base, candidate: "lower", mae_difference: -2, ci_lower: -3, ci_upper: -1 },
    ]);
    expect(many).toMatch(/4 comparisons: 1 equal, 1 practical ties, 1 unclear because the range includes zero, and 1 clear lower-error results/i);
    expect((many.match(/[.!?](?=\s|$)/g) ?? [])).toHaveLength(2);
  });

  it("keeps every interpretation short and free of unsupported claims", () => {
    render(<App release={releaseFixture} />);
    for (const region of document.querySelectorAll("[data-interpretation]")) {
      const text = region.querySelector("p").textContent;
      expect((text.match(/[.!?](?=\s|$)/g) ?? []).length).toBeGreaterThan(0);
      expect((text.match(/[.!?](?=\s|$)/g) ?? []).length).toBeLessThanOrEqual(2);
      expect(text).not.toMatch(/best|winner|caused|guaranteed|production|untouched test/i);
    }
    expect(document.body.textContent).not.toMatch(/Evaluation status|Explore the evidence|Load-Curve Atlas|state of the art|distribution-free/i);
    expect(screen.queryByRole("heading", { name: "Limitations" })).toBeNull();
    for (const limitation of releaseFixture.limitations) {
      expect(document.body.textContent).not.toContain(limitation.kind);
      expect(document.body.textContent).not.toContain(limitation.evidence);
    }
  });

  it("uses reader-facing explanations instead of internal evidence labels", () => {
    render(<App release={releaseFixture} />);

    expect(document.body.textContent).not.toMatch(/retrospective final|later-period|selected evidence|how to read the evidence/i);
    expect(screen.getByRole("combobox", { name: "Observed-value coverage target" })).toBeTruthy();
    expect(screen.getByText(/95 of every 100 observed values/i)).toBeTruthy();
    expect(screen.getByText(/does not change the point forecast/i)).toBeTruthy();
    expect(screen.getByText(/PSI compares each feature distribution/i)).toBeTruthy();
    expect(document.body.textContent).toMatch(/daily-total alignment/i);

    fireEvent.change(screen.getByRole("combobox", { name: "Language" }), { target: { value: "tr" } });
    expect(screen.getByRole("heading", { name: "Talep örüntüleri" })).toBeTruthy();
    expect(screen.getAllByRole("combobox", { name: "Sıklık" }).length).toBeGreaterThan(0);
    expect(screen.getByRole("combobox", { name: "Metrik" })).toBeTruthy();
    expect(screen.getByRole("combobox", { name: "Gerçek değer kapsama hedefi" })).toBeTruthy();
    expect(screen.getByText(/her 100 gerçek değerin yaklaşık 95/i)).toBeTruthy();
    expect(document.body.textContent).toMatch(/günlük toplam uyumu/i);
  });

  it("adds zoom only to dense charts while tables keep every selected row", () => {
    render(<App release={releaseFixture} />);

    const opening = document.querySelector(".opening-figure [data-chart-frame]");
    const tableRows = opening.querySelectorAll("tbody tr").length;
    expect(within(opening).getByRole("button", { name: "Reset zoom" })).toBeTruthy();
    expect(opening.querySelector("[data-chart-brush]")).toBeTruthy();
    fireEvent.click(within(opening).getByRole("button", { name: "Reset zoom" }));
    expect(opening.querySelectorAll("tbody tr")).toHaveLength(tableRows);

    const demand = document.querySelector("#load-patterns");
    const weekdayFigure = demand.querySelectorAll(".chapter-figure")[0];
    expect(within(weekdayFigure).queryByRole("button", { name: "Reset zoom" })).toBeNull();
  });

  it("renders the documented empty state without empty chart axes", () => {
    const emptyRelease = structuredClone(releaseFixture);
    emptyRelease.hourly_error.rows = emptyRelease.hourly_error.rows.filter((row) => row.slice_type !== "local_hour");
    render(<App release={emptyRelease} />);
    const errorFigure = document.querySelector("#reliability").querySelectorAll(".chapter-figure")[0];
    fireEvent.change(within(errorFigure).getByRole("combobox", { name: "Frequency" }), { target: { value: "hourly" } });
    fireEvent.change(within(errorFigure).getByRole("combobox", { name: "Hourly grouping" }), { target: { value: "local_hour" } });
    expect(within(errorFigure).getByRole("heading", { name: "No evidence matches these filters" })).toBeTruthy();
    expect(within(errorFigure).queryByRole("img")).toBeNull();
  });

  it("fails closed without partial claims for malformed generated evidence", () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
    render(<ErrorBoundary><App release={{ ...releaseFixture, study: undefined }} /></ErrorBoundary>);
    expect(screen.getByRole("alert").textContent).toBe("The generated evidence could not be displayed. Refresh the page and try again.");
    expect(screen.queryByRole("heading", { level: 1 })).toBeNull();
    expect(screen.queryByText("Daily comparison")).toBeNull();
    consoleError.mockRestore();
  });

  it("preserves theme, responsive, focus, print, forced-color, and static-access contracts", () => {
    render(<App release={releaseFixture} />);
    const theme = screen.getByRole("combobox", { name: "Theme" });
    fireEvent.change(theme, { target: { value: "dark" } });
    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(window.localStorage.getItem("dashboard-theme")).toBe("dark");
    fireEvent.change(theme, { target: { value: "system" } });
    expect(document.documentElement.hasAttribute("data-theme")).toBe(false);
    expect(window.localStorage.getItem("dashboard-theme")).toBeNull();

    expect(styles).toMatch(/--page:\s*#f3f5f6/i);
    expect(styles).toMatch(/--panel:\s*#fafbfb/i);
    expect(styles).toMatch(/--page:\s*#101213/i);
    expect(styles).toMatch(/--panel:\s*#171b1d/i);
    expect(styles).toMatch(/\.figure-layout\s*\{[^}]*grid-template-columns:\s*minmax\(0, 3fr\) minmax\(240px, 2fr\)/s);
    expect(styles).toMatch(/\.chart\s*\{[^}]*height:\s*210px/s);
    expect(styles).toMatch(/@media \(max-width:\s*767px\)[\s\S]*?\.chart\s*\{[^}]*height:\s*180px/s);
    expect(styles).toMatch(/@media \(forced-colors:\s*active\)/);
    expect(styles).toMatch(/@media print/);
    expect(styles).toMatch(/focus-visible[^}]*outline:\s*3px solid var\(--focus\)/s);
    expect(document.querySelectorAll(".study-chapter .chart[role='img']")).toHaveLength(10);
  });
});
