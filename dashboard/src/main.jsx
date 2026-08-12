import { Component, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";

import releaseData from "./generated/release.json";
import "./styles.css";
import { MetricSummary, StatusMessage } from "./components/Evidence.jsx";
import { LanguageControl, LanguageProvider, useI18n } from "./i18n.jsx";
import { ForecastView } from "./views/EvidenceViews.jsx";
import { StudyViews } from "./views/StudyViews.jsx";

const SECTION_LINKS = [
  ["load-patterns", "sections.demand"],
  ["forecast-performance", "sections.performance"],
  ["reliability", "sections.reliability"],
  ["evaluation-method", "sections.method"],
];

function assertRelease(release) {
  const optionKeys = [
    "frequency", "daily_models", "hourly_models", "hourly_horizons", "local_hours",
    "hourly_point_error_models", "hourly_point_error_horizons",
    "hourly_interval_coverage_horizons",
    "output_states", "daily_interval_methods", "daily_interval_levels",
    "hourly_interval_methods", "hourly_interval_levels",
  ];
  const valid =
    release?.schema_version === 1 &&
    release.final_role === "retrospective_final" &&
    typeof release.source_revision === "string" &&
    typeof release.bundle_fingerprint === "string" &&
    release.overview?.length === 3 &&
    release.forecast?.daily?.rows?.length > 0 &&
    release.forecast?.hourly?.rows?.length > 0 &&
    release.intervals?.daily?.length > 0 &&
    release.intervals?.hourly?.length > 0 &&
    release.hourly_error?.rows?.length > 0 &&
    release.hourly_error?.horizon_profile?.length > 0 &&
    release.comparison?.bootstrap?.length > 0 &&
    release.study?.load_patterns?.weekday?.length === 7 &&
    release.study?.load_patterns?.month?.length === 12 &&
    release.study?.load_patterns?.temperature?.length === 9 &&
    release.study?.load_patterns?.shap_importance?.length > 0 &&
    release.study?.performance?.daily_comparison?.length > 0 &&
    release.study?.performance?.rolling_origin_mape?.length > 0 &&
    release.study?.performance?.generalization_gap?.length > 0 &&
    release.study?.performance?.hourly_ablation?.length > 0 &&
    release.study?.reliability?.error_slices?.length > 0 &&
    release.study?.reliability?.feature_drift?.length > 0 &&
    release.study?.reliability?.residual_drift_monthly?.length > 0 &&
    Object.keys(release.study?.method?.evidence_roles ?? {}).length > 0 &&
    release.selector_options?.hourly_point_error_horizons?.length === 24 &&
    release.selector_options.hourly_point_error_horizons.every((value, index) => value === index + 1) &&
    release.selector_options?.hourly_interval_coverage_horizons?.length === 1 &&
    release.selector_options.hourly_interval_coverage_horizons[0] === 24 &&
    release.protocol?.periods?.retrospective_final?.length === 2 &&
    release.limitations?.length > 0 &&
    optionKeys.every((key) => release.selector_options?.[key]?.length > 0) &&
    release.reconciliation?.invariant?.all_passed === true;
  if (!valid) throw new Error("Invalid generated dashboard release");
}

function themeStorage() {
  try {
    return window.localStorage;
  } catch {
    return undefined;
  }
}

function ThemeControl() {
  const { t } = useI18n();
  const [theme, setTheme] = useState(() => themeStorage()?.getItem("dashboard-theme") || "system");
  useEffect(() => {
    if (theme === "system") {
      document.documentElement.removeAttribute("data-theme");
      themeStorage()?.removeItem("dashboard-theme");
    } else {
      document.documentElement.dataset.theme = theme;
      themeStorage()?.setItem("dashboard-theme", theme);
    }
  }, [theme]);
  return (
    <label className="theme-control" htmlFor="theme">
      {t("theme.label")}
      <select id="theme" value={theme} onChange={(event) => setTheme(event.target.value)}>
        <option value="system">{t("theme.system")}</option><option value="light">{t("theme.light")}</option><option value="dark">{t("theme.dark")}</option>
      </select>
    </label>
  );
}

function SiteHeader() {
  const { t } = useI18n();
  const repository = "https://github.com/frogiraffe/german-grid-load-forecasting";
  return (
    <header className="site-header">
      <strong>{t("shell.title")}</strong>
      <nav aria-label="Project resources">
        <a href={`${repository}/blob/main/docs/MODEL_CARD.md`}>{t("shell.modelCard")}</a>
        <a href={`${repository}/blob/main/report/technical-report-en.tex`}>{t("shell.report")}</a>
        <a href={repository}>{t("shell.repository")}</a>
      </nav>
      <LanguageControl />
      <ThemeControl />
    </header>
  );
}

function ReportHeader({ release }) {
  const { formatDate, t } = useI18n();
  const daily = release.forecast.daily;
  return (
    <section className="report-header">
      <h1>{t("header.title")}</h1>
      <p>{t("header.description")}</p>
      <dl>
        <div><dt>{t("header.dateRange")}</dt><dd>{formatDate(daily.start)}–{formatDate(daily.end)}</dd></div>
        <div><dt>{t("header.paths")}</dt><dd>{t("header.pathValue")}</dd></div>
      </dl>
    </section>
  );
}

function SummaryCells({ release }) {
  const { formatModel, formatMw, formatPercent, t } = useI18n();
  const [daily, hourly, intervalOverview] = release.overview;
  const interval = [...release.intervals.daily, ...release.intervals.hourly].find(
    (row) => row.method === intervalOverview.method && Number(row.nominal) === Number(intervalOverview.nominal),
  );
  const oneDay = daily.baselines.find((row) => row.model === "naive_1d");
  const sevenDay = daily.baselines.find((row) => row.model === "seasonal_naive_7d");
  if (!interval || !oneDay || !sevenDay) throw new Error("Incomplete dashboard summary evidence");
  return (
    <section className="summaries" aria-label={t("summary.label")}>
      <MetricSummary title={t("summary.daily")} accent="forecast">
        <p><strong>{t("summary.dailyMetric", { model: formatModel(daily.model), mae: formatMw(daily.metrics.MAE) })}</strong></p>
        <p>{t("summary.dailyBaselines", { oneDay: formatMw(oneDay.MAE), sevenDay: formatMw(sevenDay.MAE), n: daily.n })}</p>
      </MetricSummary>
      <MetricSummary title={t("summary.alignment")} accent="reconciliation">
        <p><strong>{t("summary.alignmentMetric", { model: formatModel(hourly.model), mae: formatMw(hourly.reconciled_mae) })}</strong></p>
        <p>{t("summary.alignmentDetail", { raw: formatMw(hourly.raw_mae), change: formatMw(hourly.mae_change), n: hourly.n })}</p>
      </MetricSummary>
      <MetricSummary title={t("summary.interval")} accent="comparison">
        <p><strong>{t("summary.intervalMethod", { method: formatModel(intervalOverview.method), level: formatPercent(intervalOverview.nominal) })}</strong></p>
        <p>{t("summary.intervalDetail", { coverage: formatPercent(intervalOverview.empirical_coverage), width: formatMw(interval.mean_width_MW ?? interval.mean_width), score: formatMw(interval.interval_score_MW ?? interval.interval_score), n: intervalOverview.n })}</p>
      </MetricSummary>
    </section>
  );
}

function SectionNav() {
  const { t } = useI18n();
  return (
    <nav className="section-nav" aria-label={t("sections.label")}>
      <div>{SECTION_LINKS.map(([id, key]) => <a key={id} href={`#${id}`}>{t(key)}</a>)}</div>
    </nav>
  );
}

export class ErrorBoundary extends Component {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() { return this.state.failed ? <main><StatusMessage type="error" /></main> : this.props.children; }
}

function Dashboard({ release }) {
  const { t } = useI18n();
  return (
    <>
      <a className="skip-link" href="#evidence">{t("shell.skip")}</a>
      <SiteHeader />
      <main id="evidence">
        <ReportHeader release={release} />
        <ForecastView release={release} overview />
        <SummaryCells release={release} />
        <SectionNav />
        <StudyViews release={release} />
      </main>
      <footer><p>{t("footer.text")}</p></footer>
    </>
  );
}

export function App({ release = releaseData }) {
  assertRelease(release);
  return <LanguageProvider><Dashboard release={release} /></LanguageProvider>;
}

const root = document.getElementById("root");
if (root) createRoot(root).render(<ErrorBoundary><App /></ErrorBoundary>);
