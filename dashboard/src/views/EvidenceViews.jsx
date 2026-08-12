import { useState } from "react";
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  ChartFrame,
  EvidenceControls,
  LiveSelection,
  ZoomBrush,
  formatCriterion,
  formatDate,
  formatModel,
  formatMw,
  formatPercent,
} from "../components/Evidence.jsx";
import { useI18n } from "../i18n.jsx";

const tooltip = { background: "var(--chart-tooltip)", borderColor: "var(--rule)", color: "var(--text)" };
const grid = { stroke: "var(--chart-grid)", strokeDasharray: "3 3" };
const axis = { stroke: "var(--chart-axis)", tick: { fill: "var(--chart-tick)" } };

const english = {
  pick: (value) => value,
  formatCriterion,
  formatDate,
  formatModel,
  formatMw,
  formatPercent,
};

export function forecastInterpretation(rows, frequency, i18n = english) {
  if (!rows.length) return "";
  const deviations = rows.map((row) => ({
    row,
    value: Math.abs(row.actual - row.forecast),
    percentage: Math.abs(row.actual - row.forecast) / Math.abs(row.actual),
  }));
  const meanPercentage = deviations.reduce((sum, item) => sum + item.percentage, 0) / deviations.length;
  const largest = deviations.reduce((maximum, item) => item.value > maximum.value ? item : maximum);
  if (largest.value === 0) return i18n.pick(
    "Forecasts match observed load across the selected period.",
    "Tahminler seçilen dönem boyunca gerçekleşen yükle aynı.",
  );
  const direction = largest.row.forecast > largest.row.actual
    ? i18n.pick("above", "üzerindeydi")
    : i18n.pick("below", "altındaydı");
  return i18n.pick(
    `The average absolute difference is ${i18n.formatPercent(meanPercentage)}. The largest miss was ${i18n.formatMw(largest.value)} ${direction} the observed load on ${i18n.formatDate(largest.row.timestamp, frequency === "hourly")}.`,
    `Ortalama mutlak fark ${i18n.formatPercent(meanPercentage)}. En büyük sapma ${i18n.formatDate(largest.row.timestamp, frequency === "hourly")} tarihinde gerçekleşen yükün ${direction} ve ${i18n.formatMw(largest.value)} büyüklüğündeydi.`,
  );
}

function DayBands({ rows, keyName }) {
  return rows.slice(0, -1).filter((_, index) => index % 2 === 0).map((row, index) => (
    <ReferenceArea key={`${row[keyName]}-${index}`} x1={row[keyName]} x2={rows[Math.min(index * 2 + 1, rows.length - 1)][keyName]} fill="var(--chart-band)" />
  ));
}

export function ForecastView({ release, overview = false, embedded = false }) {
  const i18n = useI18n();
  const { formatDate: date, formatModel: modelName, pick } = i18n;
  const selected = release.protocol.selected_models;
  const [frequency, setFrequency] = useState("daily");
  const [model, setModel] = useState(selected.daily);
  const [output, setOutput] = useState("reconciled");
  const options = release.selector_options;
  const changeFrequency = (next) => {
    setFrequency(next);
    setModel(next === "daily" ? selected.daily : selected.hourly);
  };
  const rows = frequency === "daily"
    ? release.forecast.daily.rows.map((row) => ({
        ...row,
        timestamp: row.date,
        intervalBase: row.fixed_lower_95,
        intervalWidth: row.fixed_upper_95 - row.fixed_lower_95,
      }))
    : release.forecast.hourly.rows
        .filter((row) => row.model === model && row.state === output)
        .map((row) => ({ ...row, timestamp: row.local_time }));
  const scope = frequency === "daily"
    ? `${date(release.forecast.daily.start)}–${date(release.forecast.daily.end)}`
    : `${date(release.forecast.hourly.window.local_start)}–${date(release.forecast.hourly.window.local_end)}`;
  const columns = frequency === "daily"
    ? [
        { key: "date", label: pick("Date", "Tarih"), render: (value) => date(value) },
        { key: "actual", label: pick("Observed (MW)", "Gerçekleşen (MW)") },
        { key: "forecast", label: pick("Forecast (MW)", "Tahmin (MW)") },
        { key: "fixed_lower_95", label: pick("95% lower (MW)", "%95 alt sınır (MW)") },
        { key: "fixed_upper_95", label: pick("95% upper (MW)", "%95 üst sınır (MW)") },
      ]
    : [
        { key: "local_time", label: pick("Europe/Berlin time", "Avrupa/Berlin saati"), render: (value) => date(value, true) },
        { key: "actual", label: pick("Observed (MW)", "Gerçekleşen (MW)") },
        { key: "forecast", label: `${output === "raw" ? pick("Before alignment", "Uyum öncesi") : pick("After alignment", "Uyum sonrası")} (${pick("forecast", "tahmin")}, MW)` },
        { key: "horizon", label: pick("Horizon", "Ufuk") },
      ];
  const chart = ({ zoomVersion }) => (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={rows} accessibilityLayer margin={{ top: 16, right: 20, bottom: 20, left: 16 }}>
        <CartesianGrid {...grid} />
        <DayBands rows={rows} keyName="timestamp" />
        <XAxis dataKey="timestamp" tickFormatter={(value) => frequency === "daily" ? value.slice(5) : value.slice(5, 10)} tickCount={7} {...axis} />
        <YAxis unit=" MW" width={72} {...axis} />
        <Tooltip contentStyle={tooltip} labelFormatter={(value) => frequency === "daily" ? date(value) : date(value, true)} />
        <Legend verticalAlign="top" height={36} wrapperStyle={{ color: "var(--chart-legend)" }} />
        {frequency === "daily" && <Area dataKey="intervalBase" stackId="interval" stroke="none" fill="transparent" isAnimationActive={false} />}
        {frequency === "daily" && <Area name={pick("95% interval", "%95 aralık")} dataKey="intervalWidth" stackId="interval" stroke="var(--chart-forecast)" fill="var(--chart-interval)" isAnimationActive={false} />}
        <Line name={pick("Observed", "Gerçekleşen")} dataKey="actual" stroke="var(--chart-observed)" dot={false} isAnimationActive={false} />
        <Line name={frequency === "hourly" && output === "raw" ? pick("Before alignment", "Uyum öncesi") : pick("Forecast", "Tahmin")} dataKey="forecast" stroke={output === "raw" ? "var(--chart-comparison)" : "var(--chart-forecast)"} strokeDasharray={output === "raw" ? "2 5" : "7 4"} dot={false} isAnimationActive={false} />
        <ZoomBrush rows={rows} dataKey="timestamp" version={zoomVersion} />
      </LineChart>
    </ResponsiveContainer>
  );
  const frame = (
    <ChartFrame
      title={overview ? pick("Figure 1. Observed and forecast load", "Şekil 1. Gerçekleşen ve tahmin edilen yük") : embedded ? pick("Observed and forecast load", "Gerçekleşen ve tahmin edilen yük") : pick("Forecast detail", "Tahmin ayrıntısı")}
      headingLevel={overview ? 2 : 3}
      description={pick(`${frequency} observed and forecast load for ${model}`, `${frequency === "daily" ? "günlük" : "saatlik"} gerçekleşen ve ${model} tahmini`)}
      interpretation={forecastInterpretation(rows, frequency, i18n)}
      caption={`${frequency === "daily" ? pick("Daily", "Günlük") : pick("Hourly", "Saatlik")}, ${modelName(model)}, ${pick("final evaluation", "son değerlendirme")}, ${scope}; n = ${rows.length}.`}
      columns={columns}
      rows={rows}
      zoomDataKey="timestamp"
    >{chart}</ChartFrame>
  );
  if (overview) {
    return <section className="opening-figure">{frame}</section>;
  }
  const controls = [
    { id: "forecast-frequency", label: pick("Frequency", "Sıklık"), value: frequency, onChange: changeFrequency, options: options.frequency.map((value) => ({ value, label: value === "daily" ? pick("Daily", "Günlük") : pick("Hourly", "Saatlik") })) },
    { id: "forecast-model", label: "Model", value: model, onChange: setModel, options: (frequency === "daily" ? options.daily_models : options.hourly_models).map((value) => ({ value, label: modelName(value) })) },
  ];
  if (frequency === "hourly") controls.push({ id: "forecast-output", label: pick("Hourly output", "Saatlik çıktı"), value: output, onChange: setOutput, options: options.output_states.map((value) => ({ value, label: value === "raw" ? pick("Before alignment", "Uyum öncesi") : pick("After alignment", "Uyum sonrası") })) });
  const content = (
    <>
      <EvidenceControls controls={controls} />
      <LiveSelection>{pick(`Showing the ${frequency} forecast for ${modelName(model)}, ${scope}.`, `${modelName(model)} için ${frequency === "daily" ? "günlük" : "saatlik"} tahmin gösteriliyor, ${scope}.`)}</LiveSelection>
      {frame}
    </>
  );
  if (embedded) return <div className="chapter-figure">{content}</div>;
  return <section id="forecasts" data-evidence-section aria-labelledby="forecasts-title"><h2 id="forecasts-title">{pick("Observed and forecast load", "Gerçekleşen ve tahmin edilen yük")}</h2>{content}</section>;
}

export function IntervalView({ release, embedded = false }) {
  const i18n = useI18n();
  const { formatPercent: percent, pick } = i18n;
  const options = release.selector_options;
  const [frequency, setFrequency] = useState("daily");
  const [method, setMethod] = useState("fixed");
  const [level, setLevel] = useState(options.daily_interval_levels.includes("95%") ? "95%" : options.daily_interval_levels[0]);
  const [scope, setScope] = useState("aggregate");
  const [scopeValue, setScopeValue] = useState("all");
  const changeFrequency = (next) => {
    setFrequency(next);
    setMethod(next === "daily" ? "fixed" : "symmetric");
    setLevel(next === "daily" ? (options.daily_interval_levels.includes("95%") ? "95%" : options.daily_interval_levels[0]) : options.hourly_interval_levels[0]);
    setScope("aggregate");
    setScopeValue("all");
  };
  const changeScope = (next) => {
    setScope(next);
    const values = next === "month"
      ? [...new Set(release.intervals.hourly.filter((row) => row.slice_type === "month").map((row) => row.slice_value))]
      : options.hourly_interval_coverage_horizons;
    setScopeValue(next === "aggregate" ? "all" : String(values[0]));
  };
  const source = release.intervals[frequency];
  const rows = source.filter((row) =>
    row.method === method &&
    row.level === level &&
    (frequency === "daily" || (
      row.slice_type === scope &&
      (scope === "aggregate" || String(row.slice_value) === String(scopeValue))
    )),
  ).map((row) => ({
    ...row,
    label: `${row.method} ${row.level}`,
    coverage: row.empirical_coverage * 100,
    width: row.mean_width_MW ?? row.mean_width,
    score: row.interval_score_MW ?? row.interval_score,
  }));
  const controls = [
    { id: "interval-frequency", label: pick("Frequency", "Sıklık"), value: frequency, onChange: changeFrequency, options: options.frequency.map((value) => ({ value, label: value === "daily" ? pick("Daily", "Günlük") : pick("Hourly", "Saatlik") })) },
    { id: "interval-method", label: pick("Interval method", "Aralık yöntemi"), value: method, onChange: setMethod, options: (frequency === "daily" ? options.daily_interval_methods : options.hourly_interval_methods).map((value) => ({ value, label: value[0].toUpperCase() + value.slice(1) })) },
    { id: "interval-level", label: pick("Observed-value coverage target", "Gerçek değer kapsama hedefi"), value: level, onChange: setLevel, options: frequency === "daily" ? options.daily_interval_levels : options.hourly_interval_levels },
  ];
  if (frequency === "hourly") {
    controls.push({
      id: "interval-scope",
      label: "Interval scope",
      value: scope,
      onChange: changeScope,
      options: [...new Set(source.map((row) => row.slice_type))].map((value) => ({
        value,
        label: value === "aggregate" ? "Aggregate" : value === "month" ? "Berlin-local month" : "Forecast horizon",
      })),
    });
    if (scope === "month") {
      controls.push({
        id: "interval-month",
        label: "Berlin-local month",
        value: scopeValue,
        onChange: setScopeValue,
        options: [...new Set(source.filter((row) => row.slice_type === "month").map((row) => row.slice_value))],
      });
    } else if (scope === "horizon") {
      controls.push({
        id: "interval-horizon",
        label: "Interval horizon",
        value: scopeValue,
        onChange: setScopeValue,
        options: options.hourly_interval_coverage_horizons.map((value) => ({ value, label: `h${value}` })),
      });
    }
  }
  const interval = rows[0];
  const difference = interval ? (interval.empirical_coverage - interval.nominal) * 100 : 0;
  const interpretation = interval
    ? pick(
      `The intervals contain ${percent(interval.empirical_coverage)} of observed values, compared with the ${percent(interval.nominal)} target. They ${difference < 0 ? "miss more observations than intended" : difference > 0 ? "miss fewer observations than intended" : "match the selected target"} in this sample.`,
      `Aralıklar, ${percent(interval.nominal)} hedefe karşı gerçek değerlerin ${percent(interval.empirical_coverage)} kadarını kapsıyor. Bu örneklemde ${difference < 0 ? "amaçlanandan daha fazla değer dışarıda kalıyor" : difference > 0 ? "amaçlanandan daha az değer dışarıda kalıyor" : "seçilen hedef tam olarak karşılanıyor"}.`,
    )
    : "";
  const target = Number(level.slice(0, -1));
  const targetHelp = pick(
    `${level} aims to contain about ${target} of every 100 observed values. It does not change the point forecast or prevent an extreme value from falling outside; a higher target normally needs a wider interval.`,
    `${level} hedefi, her 100 gerçek değerin yaklaşık ${target} tanesini aralıkta tutmayı amaçlar. Nokta tahminini değiştirmez ve ekstrem bir değerin dışarıda kalmasını engellemez; daha yüksek hedef genellikle daha geniş aralık gerektirir.`,
  );
  const methodHelp = {
    fixed: pick("Fixed keeps one calibration-period error margin unchanged for later forecasts.", "Fixed, kalibrasyon döneminde öğrenilen tek hata payını sonraki tahminlerde değiştirmeden kullanır."),
    symmetric: pick("Symmetric places the same error margin above and below each forecast.", "Symmetric, her tahminin altına ve üstüne aynı hata payını yerleştirir."),
    adaptive: pick("Adaptive updates the margin after actual values arrive when recent coverage is too low or too high.", "Adaptive, gerçek değerler geldikten sonra yakın dönem kapsaması düşük veya yüksekse hata payını sonraki tahminler için günceller."),
    cqr: pick("CQR calibrates lower and upper quantile forecasts separately, so the interval can be asymmetric.", "CQR, alt ve üst yüzdelik tahminlerini ayrı kalibre eder; bu nedenle aralık asimetrik olabilir."),
  }[method];
  const content = (
    <>
      <EvidenceControls controls={controls} />
      <p className="method-help">{targetHelp} {methodHelp}</p>
      <LiveSelection>{pick(`Showing ${frequency} intervals for ${method}, ${level}${frequency === "hourly" ? `, ${scope === "horizon" ? `h${scopeValue}` : scopeValue}` : ""}.`, `${method}, ${level} için ${frequency === "daily" ? "günlük" : "saatlik"} aralıklar gösteriliyor${frequency === "hourly" ? `, ${scope === "horizon" ? `h${scopeValue}` : scopeValue}` : ""}.`)}</LiveSelection>
      <ChartFrame
        title={pick("Observed-value coverage and interval width", "Gerçek değer kapsaması ve aralık genişliği")}
        description={`${frequency} ${method} interval empirical coverage`}
        interpretation={interpretation}
        caption={`${frequency === "daily" ? pick("Daily", "Günlük") : pick("Hourly", "Saatlik")}, ${method}, ${frequency === "daily" ? pick("final evaluation", "son değerlendirme") : `${scope === "month" ? pick("Berlin-local month", "Berlin yerel ayı") : scope === "horizon" ? `h${scopeValue}` : pick("aggregate", "toplam")}, ${pick("final evaluation", "son değerlendirme")}`}, ${pick("target", "hedef")} ${level}; n = ${rows[0]?.n ?? 0}.`}
        rows={rows}
        columns={[
          { key: "label", label: pick("Method and target", "Yöntem ve hedef") },
          { key: "nominal", label: pick("Coverage target", "Kapsama hedefi"), render: percent },
          { key: "empirical_coverage", label: pick("Observed-value coverage", "Gerçek değer kapsaması"), render: percent },
          { key: "width", label: pick("Mean width (MW)", "Ortalama genişlik (MW)") },
          { key: "score", label: pick("Interval score (MW)", "Aralık skoru (MW)") },
          { key: "n", label: "n" },
        ]}
      >
        <ResponsiveContainer width="100%" height="100%"><BarChart data={rows} accessibilityLayer margin={{ top: 16, right: 20, bottom: 20, left: 16 }}><CartesianGrid {...grid} /><XAxis dataKey="label" {...axis} /><YAxis unit="%" domain={[0, 100]} {...axis} /><Tooltip contentStyle={tooltip} /><Legend verticalAlign="top" height={36} wrapperStyle={{ color: "var(--chart-legend)" }} /><ReferenceLine y={target} stroke="var(--chart-reference)" strokeDasharray="6 4" label={pick("Target", "Hedef")} /><Bar name={pick("Observed-value coverage", "Gerçek değer kapsaması")} dataKey="coverage" fill="var(--chart-forecast)" isAnimationActive={false} /></BarChart></ResponsiveContainer>
      </ChartFrame>
    </>
  );
  if (embedded) return <div className="chapter-figure">{content}</div>;
  return <section id="intervals" data-evidence-section aria-labelledby="intervals-title"><h2 id="intervals-title">{pick("Prediction interval results", "Tahmin aralığı sonuçları")}</h2>{content}</section>;
}

export function HourlyErrorView({ release }) {
  const i18n = useI18n();
  const { formatModel: modelName, pick } = i18n;
  const selected = release.protocol.selected_models.hourly;
  const options = release.selector_options;
  const [model, setModel] = useState(selected);
  const [mode, setMode] = useState("horizon");
  const [slice, setSlice] = useState("all");
  const [metric, setMetric] = useState("MAE");
  const changeMode = (next) => { setMode(next); setSlice("all"); };
  const source = mode === "horizon"
    ? release.hourly_error.horizon_profile
    : release.hourly_error.rows.filter((row) => row.slice_type === "local_hour");
  const rows = source.filter((row) => row.model === model && (slice === "all" || Number(mode === "horizon" ? row.horizon : row.slice_value) === Number(slice))).map((row) => ({ ...row, label: mode === "horizon" ? `h${row.horizon}` : `${String(row.slice_value).padStart(2, "0")}:00`, value: row[metric] }));
  const sliceOptions = mode === "horizon"
    ? [{ value: "all", label: "All horizons" }, ...options.hourly_horizons.map((value) => ({ value, label: `h${value}` }))]
    : [{ value: "all", label: "All local hours" }, ...options.local_hours.map((value) => ({ value, label: `${String(value).padStart(2, "0")}:00` }))];
  const hourlyInterpretation = sliceInterpretation(rows, metric, mode, i18n);
  return (
    <section id="hourly-error" data-evidence-section aria-labelledby="hourly-error-title">
      <h2 id="hourly-error-title">Hourly error by forecast horizon and local hour</h2>
      <EvidenceControls controls={[
        { id: "hourly-model", label: "Model", value: model, onChange: setModel, options: options.hourly_models.map((value) => ({ value, label: formatModel(value) })) },
        { id: "hourly-mode", label: "Evidence mode", value: mode, onChange: changeMode, options: [{ value: "horizon", label: "Forecast horizon" }, { value: "local_hour", label: "Berlin local hour" }] },
        { id: "hourly-slice", label: mode === "horizon" ? "Forecast horizon" : "Berlin local hour", value: slice, onChange: setSlice, options: sliceOptions },
        { id: "hourly-metric", label: "Metric", value: metric, onChange: setMetric, options: ["MAE", "RMSE"] },
      ]} />
      <LiveSelection>Showing {mode === "horizon" ? "forecast horizon" : "Berlin local hour"} for {formatModel(model)}, {slice === "all" ? "all generated slices" : slice}.</LiveSelection>
      <ChartFrame
        title={`${metric} by ${mode === "horizon" ? "forecast horizon" : "Berlin local hour"}`}
        description={`Hourly ${metric} by ${mode} for ${model}`}
        interpretation={hourlyInterpretation}
        caption={`${modelName(model)}, ${pick("final evaluation", "son değerlendirme")}, ${mode === "horizon" ? pick("forecast horizon", "tahmin ufku") : pick("Berlin local hour", "Berlin yerel saati")}; n = ${rows.reduce((sum, row) => sum + row.n, 0)}.`}
        rows={rows}
        columns={[
          { key: "label", label: mode === "horizon" ? "Horizon" : "Berlin local hour" },
          { key: "MAE", label: "MAE (MW)" }, { key: "RMSE", label: "RMSE (MW)" },
          { key: "MAPE", label: "MAPE (%)" }, { key: "MASE", label: "MASE" }, { key: "n", label: "n" },
        ]}
      >
        <ResponsiveContainer width="100%" height="100%"><BarChart data={rows} accessibilityLayer margin={{ top: 16, right: 20, bottom: 20, left: 16 }}><CartesianGrid {...grid} /><XAxis dataKey="label" {...axis} /><YAxis unit=" MW" {...axis} /><Tooltip contentStyle={tooltip} /><Legend verticalAlign="top" height={36} wrapperStyle={{ color: "var(--chart-legend)" }} /><Bar name={`${metric} (MW)`} dataKey="value" fill="var(--chart-comparison)" isAnimationActive={false} /></BarChart></ResponsiveContainer>
      </ChartFrame>
    </section>
  );
}

export function sliceInterpretation(rows, metric, mode, i18n = english) {
  const minimum = rows.reduce((current, row) => row.value < current.value ? row : current, rows[0]);
  const maximum = rows.reduce((current, row) => row.value > current.value ? row : current, rows[0]);
  const constant = rows.length > 1 && rows.every((row) => row.value === rows[0].value);
  return !rows.length
    ? ""
    : rows.length === 1
    ? i18n.pick(`At ${rows[0].label}, ${metric} is ${i18n.formatMw(rows[0].value)}.`, `${rows[0].label} için ${metric} ${i18n.formatMw(rows[0].value)}.`)
    : constant
    ? i18n.pick(`Error is the same across displayed ${mode === "horizon" ? "horizons" : "local hours"}.`, `Hata, gösterilen ${mode === "horizon" ? "tahmin ufuklarında" : "yerel saatlerde"} aynı.`)
    : i18n.pick(
      `Error is lowest at ${minimum.label} and highest at ${maximum.label}, a difference of ${i18n.formatMw(maximum.value - minimum.value)}.`,
      `Hata ${minimum.label} değerinde en düşük, ${maximum.label} değerinde en yüksek; aradaki fark ${i18n.formatMw(maximum.value - minimum.value)}.`,
    );
}

export function ReconciliationView({ release }) {
  const evidence = release.reconciliation;
  const [output, setOutput] = useState("reconciled");
  const rows = [
    { id: "raw", state: "Raw", MAE: evidence.raw_mae, n: evidence.n },
    { id: "reconciled", state: "Reconciled", MAE: evidence.reconciled_mae, n: evidence.n },
  ];
  const reconciliationDirection = evidence.reconciled_mae < evidence.raw_mae
    ? "lowers"
    : evidence.reconciled_mae > evidence.raw_mae
    ? "raises"
    : "does not change";
  return (
    <section id="reconciliation" data-evidence-section aria-labelledby="reconciliation-title">
      <h2 id="reconciliation-title">Reconciliation results</h2>
      <EvidenceControls controls={[{ id: "reconciliation-output", label: "Hourly output", value: output, onChange: setOutput, options: release.selector_options.output_states.map((value) => ({ value, label: value === "raw" ? "Raw" : "Reconciled" })) }]} />
      <LiveSelection>Showing reconciliation evidence for {formatModel(evidence.model)}, {output} output.</LiveSelection>
      <ChartFrame
        title="Raw and reconciled hourly MAE"
        description={`Raw and reconciled hourly MAE for ${evidence.model}`}
        interpretation={`Reconciliation ${reconciliationDirection} MAE from ${formatMw(evidence.raw_mae)} to ${formatMw(evidence.reconciled_mae)}. Adjusted hourly totals ${evidence.invariant.all_passed ? "pass" : "fail"} the daily-consistency checks.`}
        caption={`${formatModel(evidence.model)}, final evaluation; n = ${evidence.n}.`}
        rows={rows}
        columns={[{ key: "state", label: "Output state" }, { key: "MAE", label: "MAE (MW)" }, { key: "n", label: "n" }]}
      >
        <ResponsiveContainer width="100%" height="100%"><BarChart data={rows} accessibilityLayer margin={{ top: 16, right: 20, bottom: 20, left: 16 }}><CartesianGrid {...grid} /><XAxis dataKey="state" {...axis} /><YAxis unit=" MW" {...axis} /><Tooltip contentStyle={tooltip} /><Legend verticalAlign="top" height={36} wrapperStyle={{ color: "var(--chart-legend)" }} /><Bar name="MAE (MW)" dataKey="MAE" fill="var(--chart-reconciliation)" isAnimationActive={false} /></BarChart></ResponsiveContainer>
      </ChartFrame>
      <p className="invariant-row">Invariant: {evidence.invariant.pass_count} passed across {evidence.invariant.complete_local_days} complete local days; maximum absolute delta {evidence.invariant.max_abs_delta}, tolerance {evidence.invariant.tolerance}.</p>
    </section>
  );
}

export function comparisonInterpretation(rows, i18n = english) {
  const describeComparison = (row) => {
    const candidate = i18n.formatModel(row.candidate);
    const reference = i18n.formatModel(row.reference);
    const includesZero = row.ci_lower <= 0 && row.ci_upper >= 0;
    if (row.mae_difference === 0 && row.ci_lower === 0 && row.ci_upper === 0) {
      return i18n.pick(`${candidate} and ${reference} have equal MAE in the generated comparison`, `${candidate} ve ${reference} aynı MAE değerine sahip`);
    }
    if (row.practical_tie) {
      return i18n.pick(`${candidate} and ${reference} are practically tied${includesZero ? " because the comparison range includes zero" : ""}`, `${candidate} ve ${reference} pratikte eşit${includesZero ? "; çünkü karşılaştırma aralığı sıfırı içeriyor" : ""}`);
    }
    if (includesZero) {
      return `${candidate} and ${reference} have no clear lower-error result because the comparison range includes zero`;
    }
    const candidateIsLower = row.ci_upper < 0;
    const lowerModel = candidateIsLower ? candidate : reference;
    const higherModel = candidateIsLower ? reference : candidate;
    return `${lowerModel} has lower MAE than ${higherModel} and its comparison range stays ${candidateIsLower ? "below" : "above"} zero`;
  };
  const equal = rows.filter((row) => row.mae_difference === 0 && row.ci_lower === 0 && row.ci_upper === 0).length;
  const practicalTies = rows.filter((row) => row.practical_tie && !(row.mae_difference === 0 && row.ci_lower === 0 && row.ci_upper === 0)).length;
  const unclear = rows.filter((row) => !row.practical_tie && !(row.mae_difference === 0 && row.ci_lower === 0 && row.ci_upper === 0) && row.ci_lower <= 0 && row.ci_upper >= 0).length;
  return rows.length > 2
    ? `Across ${rows.length} comparisons: ${equal} equal, ${practicalTies} practical ties, ${unclear} unclear because the range includes zero, and ${rows.length - equal - practicalTies - unclear} clear lower-error results. Exact model-by-model results are in the table.`
    : `${rows.map(describeComparison).join("; ")}.`;
}

export function ComparisonView({ release }) {
  const selection = release.comparison.selection;
  const rows = release.comparison.bootstrap.map((row, index) => ({ ...row, id: index, comparison: `${formatModel(row.candidate)} vs ${formatModel(row.reference)}` }));
  const decision = release.comparison.low_risk_decision;
  const interpretation = comparisonInterpretation(rows);
  return (
    <section id="comparison" data-evidence-section aria-labelledby="comparison-title">
      <h2 id="comparison-title">Model comparison and selection</h2>
      <p>Validation selection criterion: <code>{formatCriterion(selection.selection_metric)}</code>. Selected model: <strong>{formatModel(selection.selected_model)}</strong>.</p>
      <ChartFrame
        title="Candidate versus reference MAE difference"
        description="Paired local-day bootstrap MAE difference and zero reference"
        interpretation={interpretation}
        caption={`Final evaluation with paired local-day samples; n = ${rows[0]?.n_days ?? 0} days.`}
        rows={rows}
        columns={[
          { key: "comparison", label: "Comparison" }, { key: "mae_difference", label: "MAE difference (MW)" },
          { key: "ci_lower", label: "CI lower (MW)" }, { key: "ci_upper", label: "CI upper (MW)" },
          { key: "practical_tie", label: "Practical tie", render: (value) => value ? "Yes" : "No" }, { key: "n_days", label: "n days" },
        ]}
      >
        <ResponsiveContainer width="100%" height="100%"><BarChart data={rows} accessibilityLayer margin={{ top: 16, right: 20, bottom: 20, left: 16 }}><CartesianGrid {...grid} /><XAxis dataKey="comparison" {...axis} /><YAxis unit=" MW" {...axis} /><Tooltip contentStyle={tooltip} /><Legend verticalAlign="top" height={36} wrapperStyle={{ color: "var(--chart-legend)" }} /><ReferenceLine y={0} stroke="var(--chart-reference)" /><Bar name="MAE difference (MW)" dataKey="mae_difference" fill="var(--chart-comparison)" isAnimationActive={false} /></BarChart></ResponsiveContainer>
      </ChartFrame>
      <p className="decision-record">Low-risk decision: <strong>{decision.decision === "accepted" ? "Accepted" : "Rejected"}</strong>. Criterion <code>{formatCriterion(decision.criterion)}</code>; {decision.rationale}.</p>
    </section>
  );
}

function periodDays([start, end]) {
  return Math.round((Date.parse(`${end}T00:00:00Z`) - Date.parse(`${start}T00:00:00Z`)) / 86400000) + 1;
}

export function ProtocolView({ release, embedded = false }) {
  const i18n = useI18n();
  const { formatDate: date, pick } = i18n;
  const labels = { train: pick("Training", "Eğitim"), validation: pick("Validation", "Doğrulama"), calibration: pick("Calibration", "Kalibrasyon"), retrospective_final: pick("Final evaluation", "Son değerlendirme") };
  const rows = ["train", "validation", "calibration", "retrospective_final"].map((key) => ({
    id: key, period: labels[key], start: release.protocol.periods[key][0], end: release.protocol.periods[key][1], days: periodDays(release.protocol.periods[key]),
  }));
  const frame = (
      <ChartFrame
        title={pick("Evaluation periods", "Değerlendirme dönemleri")}
        description="Chronological training, validation, calibration, and final evaluation periods"
        interpretation={pick(
          "Training fits the models, validation compares choices, calibration sets interval widths, and final evaluation measures the fixed result. Temperature-bin averages also contain season, weekday, holiday, daylight and economic differences, so they do not isolate temperature's MW effect.",
          "Eğitim modelleri kurar, doğrulama seçenekleri karşılaştırır, kalibrasyon aralık genişliklerini belirler ve son değerlendirme sabit sonucu ölçer. Sıcaklık grubu ortalamaları mevsim, hafta günü, tatil, gün ışığı ve ekonomik farkları da içerdiği için sıcaklığın tek başına MW etkisini ayırmaz.",
        )}
        caption={`${pick("Chronological evaluation design", "Kronolojik değerlendirme düzeni")}; n = ${rows.reduce((sum, row) => sum + row.days, 0)} ${pick("dated observations", "tarihli gözlem")}.`}
        rows={rows}
        columns={[
          { key: "period", label: pick("Period", "Dönem") }, { key: "start", label: pick("Start", "Başlangıç"), render: (value) => date(value) },
          { key: "end", label: pick("End", "Bitiş"), render: (value) => date(value) }, { key: "days", label: pick("Calendar days", "Takvim günü") },
        ]}
      >
        <ResponsiveContainer width="100%" height="100%"><BarChart data={rows} accessibilityLayer layout="vertical" margin={{ top: 16, right: 20, bottom: 20, left: 24 }}><CartesianGrid {...grid} /><XAxis type="number" unit=" days" {...axis} /><YAxis type="category" dataKey="period" width={112} {...axis} /><Tooltip contentStyle={tooltip} /><Legend verticalAlign="top" height={36} wrapperStyle={{ color: "var(--chart-legend)" }} /><Bar name="Calendar days" dataKey="days" fill="var(--chart-reference)" isAnimationActive={false} /></BarChart></ResponsiveContainer>
      </ChartFrame>
  );
  if (embedded) return <div className="chapter-figure">{frame}</div>;
  return <section id="protocol" data-evidence-section aria-labelledby="protocol-title"><h2 id="protocol-title">Evaluation protocol</h2>{frame}</section>;
}

export function EvidenceViews({ release }) {
  return <><ForecastView release={release} /><IntervalView release={release} /><HourlyErrorView release={release} /><ReconciliationView release={release} /><ComparisonView release={release} /><ProtocolView release={release} /></>;
}
