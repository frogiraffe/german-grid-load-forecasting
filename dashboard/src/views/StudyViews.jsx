import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
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
} from "../components/Evidence.jsx";
import { useI18n } from "../i18n.jsx";
import {
  ForecastView,
  IntervalView,
  ProtocolView,
  comparisonInterpretation,
  sliceInterpretation,
} from "./EvidenceViews.jsx";

const tooltip = { background: "var(--chart-tooltip)", borderColor: "var(--rule)", color: "var(--text)" };
const grid = { stroke: "var(--chart-grid)", strokeDasharray: "3 3" };
const axis = { stroke: "var(--chart-axis)", tick: { fill: "var(--chart-tick)" } };

function unique(values) {
  return [...new Set(values)];
}

function roleLabel(value, { formatModel, pick }) {
  return {
    full_available_history_descriptive: pick("Available-history summary", "Mevcut geçmiş özeti"),
    retrospective_explanation: pick("Study explanation", "Çalışma açıklaması"),
    retrospective_final: pick("Final evaluation", "Son değerlendirme"),
    validation: pick("Validation", "Doğrulama"),
  }[value] ?? formatModel(value);
}

function rangeInterpretation(rows, noun, i18n, valueKey = "value", format = i18n.formatMw) {
  if (!rows.length) return "";
  if (rows.length === 1) return i18n.pick(`${noun} is ${format(rows[0][valueKey])} for ${rows[0].label}.`, `${rows[0].label} için ${noun.toLocaleLowerCase("tr-TR")} ${format(rows[0][valueKey])}.`);
  if (rows.every((row) => row[valueKey] === rows[0][valueKey])) {
    return i18n.pick(`${noun} is the same across the displayed groups.`, `${noun} gösterilen gruplarda aynı.`);
  }
  const minimum = rows.reduce((current, row) => row[valueKey] < current[valueKey] ? row : current);
  const maximum = rows.reduce((current, row) => row[valueKey] > current[valueKey] ? row : current);
  const difference = maximum[valueKey] - minimum[valueKey];
  return i18n.pick(
    `${noun} is lowest for ${minimum.label} and highest for ${maximum.label}, a difference of ${format(difference)}.`,
    `${noun}, ${minimum.label} grubunda en düşük; ${maximum.label} grubunda en yüksek. Aradaki fark ${format(difference)}.`,
  );
}

function LoadProfileFigure({ release }) {
  const i18n = useI18n();
  const { pick } = i18n;
  const [profile, setProfile] = useState("weekday");
  const rows = release.study.load_patterns[profile].map((row) => ({
    ...row,
    label: profile === "weekday" ? row.weekday : row.month,
    value: row.mean_load_MW,
  }));
  return (
    <div className="chapter-figure">
      <EvidenceControls controls={[{
        id: "load-profile",
        label: pick("Demand grouping", "Talep gruplaması"),
        value: profile,
        onChange: setProfile,
        options: [{ value: "weekday", label: pick("Weekday", "Haftanın günü") }, { value: "month", label: pick("Month", "Ay") }],
      }]} />
      <LiveSelection>{pick(`Showing average demand by ${profile}.`, profile === "weekday" ? "Haftanın gününe göre ortalama talep gösteriliyor." : "Aya göre ortalama talep gösteriliyor.")}</LiveSelection>
      <ChartFrame
        title={pick("Average demand pattern", "Ortalama talep örüntüsü")}
        description={`Mean daily load by ${profile}`}
        interpretation={rangeInterpretation(rows, pick("Mean load", "Ortalama yük"), i18n)}
        caption={`${pick("Average by", "Gruplama")}: ${profile === "weekday" ? pick("weekday", "haftanın günü") : pick("month", "ay")}; n = ${rows.reduce((sum, row) => sum + row.n_days, 0)} ${pick("days", "gün")}.`}
        rows={rows}
        columns={[
          { key: "label", label: profile === "weekday" ? pick("Weekday", "Haftanın günü") : pick("Month", "Ay") },
          { key: "mean_load_MW", label: pick("Mean load (MW)", "Ortalama yük (MW)") },
          { key: "n_days", label: pick("n days", "gün sayısı") },
        ]}
      >
        <ResponsiveContainer width="100%" height="100%"><BarChart data={rows} accessibilityLayer margin={{ top: 16, right: 20, bottom: 20, left: 16 }}><CartesianGrid {...grid} /><XAxis dataKey="label" {...axis} /><YAxis unit=" MW" width={72} {...axis} /><Tooltip contentStyle={tooltip} /><Legend verticalAlign="top" height={36} wrapperStyle={{ color: "var(--chart-legend)" }} /><Bar name={pick("Mean load (MW)", "Ortalama yük (MW)")} dataKey="value" fill="var(--chart-reference)" isAnimationActive={false} /></BarChart></ResponsiveContainer>
      </ChartFrame>
    </div>
  );
}

function TemperatureFigure({ release }) {
  const i18n = useI18n();
  const { pick } = i18n;
  const rows = release.study.load_patterns.temperature.map((row) => ({
    ...row,
    label: pick(`${row.lower_C} to <${row.upper_C} °C`, `${row.lower_C}–<${row.upper_C} °C`),
    value: row.mean_load_MW,
  }));
  return (
    <div className="chapter-figure">
      <ChartFrame
        title={pick("Temperature and demand", "Sıcaklık ve talep")}
        description="Mean daily load by forecast-origin temperature bin"
        interpretation={pick(
          `${rangeInterpretation(rows, "Mean load", i18n)} Heating and cooling are plausible reasons for this demand relationship.`,
          `${rangeInterpretation(rows, "Ortalama yük", i18n)} Isıtma ve soğutma, bu talep ilişkisinin olası nedenleridir.`,
        )}
        caption={`${pick("Forecast-time temperature groups", "Tahmin anındaki sıcaklık grupları")}; n = ${rows.reduce((sum, row) => sum + row.n_days, 0)} ${pick("days", "gün")}.`}
        rows={rows}
        columns={[
          { key: "label", label: pick("Temperature range", "Sıcaklık aralığı") },
          { key: "mean_load_MW", label: pick("Mean load (MW)", "Ortalama yük (MW)") },
          { key: "n_days", label: pick("n days", "gün sayısı") },
        ]}
      >
        <ResponsiveContainer width="100%" height="100%"><LineChart data={rows} accessibilityLayer margin={{ top: 16, right: 20, bottom: 20, left: 16 }}><CartesianGrid {...grid} /><XAxis dataKey="label" {...axis} /><YAxis unit=" MW" width={72} {...axis} /><Tooltip contentStyle={tooltip} /><Legend verticalAlign="top" height={36} wrapperStyle={{ color: "var(--chart-legend)" }} /><Line name={pick("Mean load (MW)", "Ortalama yük (MW)")} dataKey="value" stroke="var(--chart-reference)" dot isAnimationActive={false} /></LineChart></ResponsiveContainer>
      </ChartFrame>
    </div>
  );
}

function ShapFigure({ release }) {
  const i18n = useI18n();
  const { formatModel, pick } = i18n;
  const source = release.study.load_patterns.shap_importance;
  const models = ["mean", "xgboost", "lightgbm"].filter((key) => source.every((row) => Number.isFinite(Number(row[key]))));
  const [model, setModel] = useState(models[0]);
  const rows = source.map((row) => ({ ...row, label: row.feature, value: row[model] }));
  const leading = rows.reduce((current, row) => row.value > current.value ? row : current, rows[0]);
  return (
    <div className="chapter-figure">
      <EvidenceControls controls={[{
        id: "shap-model",
        label: pick("Model", "Model"),
        value: model,
        onChange: setModel,
        options: models.map((value) => ({
          value,
          label: value === "mean" ? "Mean" : value === "xgboost" ? "XGBoost" : "LightGBM",
        })),
      }]} />
      <LiveSelection>{pick(`Showing model signal importance for ${model === "mean" ? "the model average" : formatModel(model)}.`, `${model === "mean" ? "Model ortalaması" : formatModel(model)} için model sinyali önemi gösteriliyor.`)}</LiveSelection>
      <ChartFrame
        title={pick("Model signal importance", "Model sinyali önemi")}
        description={`SHAP importance for ${model}`}
        interpretation={pick(
          `${leading.label} is the signal used most strongly by the displayed model. Mean absolute SHAP measures strength of use, not whether the signal raises or lowers demand.`,
          `${leading.label}, gösterilen modelin en güçlü kullandığı sinyal. Ortalama mutlak SHAP, kullanım gücünü ölçer; sinyalin talebi artırdığını veya azalttığını göstermez.`,
        )}
        caption={`${pick("Tree-model signal summary", "Ağaç modeli sinyal özeti")}; n = ${rows.length} ${pick("features", "özellik")}.`}
        rows={rows}
        columns={[{ key: "feature", label: pick("Feature", "Özellik") }, { key: model, label: pick("Mean absolute SHAP importance", "Ortalama mutlak SHAP önemi") }]}
      >
        <ResponsiveContainer width="100%" height="100%"><BarChart data={rows} accessibilityLayer layout="vertical" margin={{ top: 16, right: 20, bottom: 20, left: 24 }}><CartesianGrid {...grid} /><XAxis type="number" {...axis} /><YAxis type="category" dataKey="label" width={112} {...axis} /><Tooltip contentStyle={tooltip} /><Legend verticalAlign="top" height={36} wrapperStyle={{ color: "var(--chart-legend)" }} /><Bar name="Mean absolute SHAP importance" dataKey="value" fill="var(--chart-comparison)" isAnimationActive={false} /></BarChart></ResponsiveContainer>
      </ChartFrame>
    </div>
  );
}

function performanceInterpretation(rows, i18n) {
  const { formatModel, formatMw, pick } = i18n;
  if (!rows.length) return "";
  if (rows.length === 1) return pick(`${formatModel(rows[0].model)} has MAE ${formatMw(rows[0].value)}.`, `${formatModel(rows[0].model)} için MAE ${formatMw(rows[0].value)}.`);
  const ordered = [...rows].sort((left, right) => left.value - right.value);
  if (ordered.every((row) => row.value === ordered[0].value)) return pick(`All ${rows.length} displayed models have equal MAE.`, `Gösterilen ${rows.length} modelin MAE değeri aynı.`);
  return pick(
    `${formatModel(ordered[0].model)} has the lowest displayed MAE, ${formatMw(ordered[1].value - ordered[0].value)} below the next model.`,
    `${formatModel(ordered[0].model)} gösterilen en düşük MAE değerine sahip; sonraki modelden ${formatMw(ordered[1].value - ordered[0].value)} daha düşük.`,
  );
}

function ModelComparisonFigure({ release }) {
  const i18n = useI18n();
  const { formatModel, pick } = i18n;
  const daily = release.study.performance.daily_comparison;
  const hourly = release.study.performance.hourly_ablation;
  const [frequency, setFrequency] = useState("daily");
  const [role, setRole] = useState("retrospective_final");
  const [state, setState] = useState("reconciled");
  const [comparisonMode, setComparisonMode] = useState("models");
  const changeFrequency = (next) => {
    setFrequency(next);
    setRole(next === "daily" ? "retrospective_final" : "validation");
    setState("reconciled");
    setComparisonMode("models");
  };
  const rows = comparisonMode === "uncertainty"
    ? release.comparison.bootstrap.map((row, index) => ({
        ...row,
        id: index,
        label: `${formatModel(row.candidate)} vs ${formatModel(row.reference)}`,
        value: row.mae_difference,
        n_value: row.n_days,
      }))
    : (frequency === "daily"
    ? daily.map((row) => ({ ...row, value: row.MAE, n_value: row.n }))
    : hourly.filter((row) => row.evaluation_period === role && row.state === state).map((row) => ({ ...row, value: row.hourly_MAE, n_value: row.n_hours })))
    .map((row) => ({ ...row, label: formatModel(row.model) }));
  const controls = [{
    id: "comparison-frequency", label: pick("Frequency", "Sıklık"), value: frequency, onChange: changeFrequency,
    options: release.selector_options.frequency.map((value) => ({ value, label: value === "daily" ? pick("Daily", "Günlük") : pick("Hourly", "Saatlik") })),
  }];
  if (frequency === "hourly") {
    controls.push({ id: "comparison-mode", label: pick("Comparison view", "Karşılaştırma görünümü"), value: comparisonMode, onChange: setComparisonMode, options: [{ value: "models", label: pick("Model error", "Model hatası") }, { value: "uncertainty", label: pick("Paired comparison range", "Eşleştirilmiş karşılaştırma aralığı") }] });
    if (comparisonMode === "models") controls.push(
      { id: "comparison-role", label: pick("Evaluation period", "Değerlendirme dönemi"), value: role, onChange: setRole, options: unique(hourly.map((row) => row.evaluation_period)).map((value) => ({ value, label: roleLabel(value, i18n) })) },
      { id: "comparison-state", label: pick("Hourly output", "Saatlik çıktı"), value: state, onChange: setState, options: unique(hourly.map((row) => row.state)).map((value) => ({ value, label: value === "raw" ? pick("Before alignment", "Uyum öncesi") : pick("After alignment", "Uyum sonrası") })) },
    );
  }
  const interpretation = comparisonMode === "uncertainty" ? comparisonInterpretation(rows, i18n) : performanceInterpretation(rows, i18n);
  const evidenceRole = comparisonMode === "uncertainty" ? roleLabel("retrospective_final", i18n) : roleLabel(role, i18n);
  return (
    <div className="chapter-figure">
      <EvidenceControls controls={controls} />
      <LiveSelection>{pick(
        `Showing ${frequency} ${comparisonMode === "uncertainty" ? "paired comparison ranges" : "model errors"}, ${evidenceRole}${frequency === "hourly" && comparisonMode === "models" ? `, ${state}` : ""}.`,
        `${frequency === "daily" ? "Günlük" : "Saatlik"} ${comparisonMode === "uncertainty" ? "eşleştirilmiş karşılaştırma aralıkları" : "model hataları"} gösteriliyor; ${evidenceRole}.`,
      )}</LiveSelection>
      <ChartFrame
        title={pick("Model and baseline comparison", "Model ve temel yöntem karşılaştırması")}
        description={`${frequency} ${comparisonMode === "uncertainty" ? "paired model uncertainty" : "model MAE comparison"}`}
        interpretation={interpretation}
        caption={`${frequency === "daily" ? "Daily" : "Hourly"}, ${evidenceRole}${frequency === "hourly" && comparisonMode === "models" ? `, ${state}` : ""}; n = ${Math.max(...rows.map((row) => row.n_value), 0)}.`}
        rows={rows}
        columns={comparisonMode === "uncertainty" ? [
          { key: "label", label: pick("Comparison", "Karşılaştırma") },
          { key: "mae_difference", label: pick("MAE difference (MW)", "MAE farkı (MW)") },
          { key: "ci_lower", label: pick("Lower bound (MW)", "Alt sınır (MW)") },
          { key: "ci_upper", label: pick("Upper bound (MW)", "Üst sınır (MW)") },
          { key: "practical_tie", label: pick("Practical tie", "Pratik eşitlik"), render: (value) => value ? pick("Yes", "Evet") : pick("No", "Hayır") },
          { key: "n_days", label: pick("n days", "gün sayısı") },
        ] : [{ key: "label", label: "Model" }, { key: "value", label: "MAE (MW)" }, { key: "n_value", label: "n" }]}
      >
        <ResponsiveContainer width="100%" height="100%"><BarChart data={rows} accessibilityLayer layout="vertical" margin={{ top: 16, right: 20, bottom: 20, left: 16 }}><CartesianGrid {...grid} /><XAxis type="number" unit=" MW" {...axis} /><YAxis type="category" dataKey="label" width={156} interval={0} {...axis} /><Tooltip contentStyle={tooltip} /><Legend verticalAlign="top" height={36} wrapperStyle={{ color: "var(--chart-legend)" }} />{comparisonMode === "uncertainty" && <ReferenceLine x={0} stroke="var(--chart-reference)" />}<Bar name={comparisonMode === "uncertainty" ? "MAE difference (MW)" : "MAE (MW)"} dataKey="value" fill="var(--chart-comparison)" isAnimationActive={false} /></BarChart></ResponsiveContainer>
      </ChartFrame>
    </div>
  );
}

function PerformanceContextFigure({ release }) {
  const i18n = useI18n();
  const { formatModel, pick } = i18n;
  const evidence = release.study.performance;
  const [mode, setMode] = useState("rolling");
  const rows = mode === "rolling"
    ? evidence.rolling_origin_mape.map((row) => ({ ...row, label: `${row.period} / ${formatModel(row.model)}`, value: row.MAPE }))
    : evidence.generalization_gap.flatMap((row) => [
        { model: row.model, label: `${formatModel(row.model)} / Validation`, period: "Validation", value: row.validation_MAPE },
        { model: row.model, label: `${formatModel(row.model)} / ${pick("Final evaluation", "Son değerlendirme")}`, period: pick("Final evaluation", "Son değerlendirme"), value: row.retrospective_final_MAPE },
      ]);
  const values = rows.map((row) => row.value);
  const interpretation = mode === "rolling"
    ? pick(
      `Across rolling validation windows, MAPE ranges from ${Math.min(...values).toFixed(2)}% to ${Math.max(...values).toFixed(2)}%. A smaller range means performance changed less between windows.`,
      `Kayan doğrulama pencerelerinde MAPE %${Math.min(...values).toFixed(2)} ile %${Math.max(...values).toFixed(2)} arasında. Daha dar aralık, performansın pencereler arasında daha az değiştiğini gösterir.`,
    )
    : pick(
      `${evidence.generalization_gap.filter((row) => row.retrospective_final_MAPE < row.validation_MAPE).length} of ${evidence.generalization_gap.length} models have lower MAPE in final evaluation than in validation. The paired bars show how each model's error changed between periods.`,
      `${evidence.generalization_gap.length} modelin ${evidence.generalization_gap.filter((row) => row.retrospective_final_MAPE < row.validation_MAPE).length} tanesinde son değerlendirme MAPE değeri doğrulamadan daha düşük. Yan yana çubuklar her modelin hatasının dönemler arasında nasıl değiştiğini gösterir.`,
    );
  return (
    <div className="chapter-figure">
      <EvidenceControls controls={[{
        id: "performance-context",
        label: pick("Performance view", "Performans görünümü"),
        value: mode,
        onChange: setMode,
        options: [
          { value: "rolling", label: pick("Validation stability", "Doğrulama kararlılığı") },
          { value: "generalization", label: pick("Validation vs final evaluation", "Doğrulama ve son değerlendirme") },
        ],
      }]} />
      <LiveSelection>{mode === "rolling" ? pick("Showing validation stability across rolling windows.", "Kayan pencerelerde doğrulama kararlılığı gösteriliyor.") : pick("Showing validation and final-evaluation error side by side.", "Doğrulama ve son değerlendirme hatası yan yana gösteriliyor.")}</LiveSelection>
      <ChartFrame
        title={mode === "rolling" ? pick("Validation stability", "Doğrulama kararlılığı") : pick("Validation vs final evaluation", "Doğrulama ve son değerlendirme")}
        description="Validation stability and retrospective performance context"
        interpretation={interpretation}
        caption={`${mode === "rolling" ? pick("Validation windows", "Doğrulama pencereleri") : pick("Validation and final evaluation", "Doğrulama ve son değerlendirme")}; n = ${rows.length}.`}
        rows={rows}
        columns={[{ key: "label", label: pick("Model and period", "Model ve dönem") }, { key: "value", label: "MAPE (%)" }]}
        zoomDataKey="label"
      >
        {({ zoomVersion }) => <ResponsiveContainer width="100%" height="100%"><BarChart data={rows} accessibilityLayer margin={{ top: 16, right: 20, bottom: 20, left: 16 }}><CartesianGrid {...grid} /><XAxis dataKey="label" {...axis} /><YAxis unit="%" {...axis} /><Tooltip contentStyle={tooltip} /><Legend verticalAlign="top" height={36} wrapperStyle={{ color: "var(--chart-legend)" }} /><Bar name="MAPE (%)" dataKey="value" fill="var(--chart-reference)" isAnimationActive={false} /><ZoomBrush rows={rows} dataKey="label" version={zoomVersion} /></BarChart></ResponsiveContainer>}
      </ChartFrame>
    </div>
  );
}

function ErrorSlicesFigure({ release }) {
  const i18n = useI18n();
  const { formatModel, formatMw, pick } = i18n;
  const daily = release.study.reliability.error_slices;
  const options = release.selector_options;
  const [frequency, setFrequency] = useState("daily");
  const [role, setRole] = useState("retrospective_final");
  const [model, setModel] = useState(release.protocol.selected_models.daily);
  const [mode, setMode] = useState("month");
  const [slice, setSlice] = useState("all");
  const [metric, setMetric] = useState("MAE");
  const changeFrequency = (next) => {
    setFrequency(next);
    setRole("retrospective_final");
    setMode(next === "daily" ? "month" : "horizon");
    setModel(next === "daily" ? release.protocol.selected_models.daily : options.hourly_point_error_models[0]);
    setSlice("all");
  };
  const changeMode = (next) => {
    setMode(next);
    const source = next === "horizon" ? release.hourly_error.horizon_profile : release.hourly_error.rows.filter((row) => row.slice_type === "local_hour");
    const models = unique(source.map((row) => row.model));
    if (!models.includes(model)) setModel(models[0]);
    setSlice("all");
  };
  const source = frequency === "daily"
    ? daily.filter((row) => row.evaluation_period === role && row.model === model && row.slice_type === mode).map((row) => ({ ...row, label: row.slice }))
    : (mode === "horizon" ? release.hourly_error.horizon_profile : release.hourly_error.rows.filter((row) => row.slice_type === "local_hour"))
        .filter((row) => row.model === model)
        .map((row) => ({ ...row, label: mode === "horizon" ? `h${row.horizon}` : `${String(row.slice_value).padStart(2, "0")}:00` }));
  const rows = source.filter((row) => slice === "all" || row.label === slice).map((row) => ({ ...row, value: row[metric] }));
  const controls = [{ id: "slice-frequency", label: pick("Frequency", "Sıklık"), value: frequency, onChange: changeFrequency, options: options.frequency.map((value) => ({ value, label: value === "daily" ? pick("Daily", "Günlük") : pick("Hourly", "Saatlik") })) }];
  if (frequency === "daily") controls.push(
    { id: "slice-role", label: pick("Evaluation period", "Değerlendirme dönemi"), value: role, onChange: setRole, options: unique(daily.map((row) => row.evaluation_period)).map((value) => ({ value, label: roleLabel(value, i18n) })) },
    { id: "slice-model", label: "Model", value: model, onChange: setModel, options: unique(daily.map((row) => row.model)).map((value) => ({ value, label: formatModel(value) })) },
    { id: "slice-mode", label: pick("Grouping", "Gruplama"), value: mode, onChange: (next) => { setMode(next); setSlice("all"); }, options: unique(daily.map((row) => row.slice_type)).map((value) => ({ value, label: value === "day_type" ? pick("Day type", "Gün türü") : pick("Month", "Ay") })) },
  );
  else controls.push(
    { id: "slice-mode", label: pick("Hourly grouping", "Saatlik gruplama"), value: mode, onChange: changeMode, options: [{ value: "horizon", label: pick("Forecast horizon", "Tahmin ufku") }, { value: "local_hour", label: pick("Berlin local hour", "Berlin yerel saati") }] },
    { id: "slice-model", label: "Model", value: model, onChange: setModel, options: unique((mode === "horizon" ? release.hourly_error.horizon_profile : release.hourly_error.rows.filter((row) => row.slice_type === "local_hour")).map((row) => row.model)).map((value) => ({ value, label: formatModel(value) })) },
    { id: "slice-value", label: mode === "horizon" ? pick("Forecast horizon", "Tahmin ufku") : pick("Berlin local hour", "Berlin yerel saati"), value: slice, onChange: setSlice, options: [{ value: "all", label: mode === "horizon" ? pick("All horizons", "Tüm ufuklar") : pick("All local hours", "Tüm yerel saatler") }, ...(mode === "horizon" ? options.hourly_point_error_horizons.map((value) => ({ value: `h${value}`, label: `h${value}` })) : options.local_hours.map((value) => ({ value: `${String(value).padStart(2, "0")}:00`, label: `${String(value).padStart(2, "0")}:00` })))] },
  );
  controls.push({ id: "slice-metric", label: pick("Metric", "Metrik"), value: metric, onChange: setMetric, options: frequency === "daily" ? ["MAE", "MAPE"] : ["MAE", "RMSE"] });
  const interpretation = frequency === "hourly"
    ? sliceInterpretation(rows, metric, mode)
    : rangeInterpretation(rows, pick(`${metric} error`, `${metric} hatası`), i18n, "value", metric === "MAPE" ? (value) => `${Number(value).toFixed(2)}%` : formatMw);
  return (
    <div className="chapter-figure">
      <EvidenceControls controls={controls} />
      <LiveSelection>{pick(`Showing ${frequency} ${formatModel(mode)} error for ${formatModel(model)}.`, `${formatModel(model)} için ${frequency === "daily" ? "günlük" : "saatlik"} hata grupları gösteriliyor.`)}</LiveSelection>
      <ChartFrame
        title={pick("Error by group", "Gruplara göre hata")}
        description={`${frequency} ${metric} error by ${mode}`}
        interpretation={interpretation}
        caption={`${frequency === "daily" ? roleLabel(role, i18n) : roleLabel("retrospective_final", i18n)}, ${formatModel(model)}, ${formatModel(mode)}; n = ${rows.reduce((sum, row) => sum + Number(row.n), 0)}.`}
        rows={rows}
        columns={[{ key: "label", label: formatModel(mode) }, { key: metric, label: metric === "MAPE" ? "MAPE (%)" : `${metric} (MW)` }, { key: "n", label: "n" }]}
        zoomDataKey="label"
      >
        {({ zoomVersion }) => <ResponsiveContainer width="100%" height="100%"><BarChart data={rows} accessibilityLayer margin={{ top: 16, right: 20, bottom: 20, left: 16 }}><CartesianGrid {...grid} /><XAxis dataKey="label" {...axis} /><YAxis unit={metric === "MAPE" ? "%" : " MW"} width={72} {...axis} /><Tooltip contentStyle={tooltip} /><Legend verticalAlign="top" height={36} wrapperStyle={{ color: "var(--chart-legend)" }} /><Bar name={metric === "MAPE" ? "MAPE (%)" : `${metric} (MW)`} dataKey="value" fill="var(--chart-comparison)" isAnimationActive={false} /><ZoomBrush rows={rows} dataKey="label" version={zoomVersion} /></BarChart></ResponsiveContainer>}
      </ChartFrame>
    </div>
  );
}

function DriftFigure({ release }) {
  const i18n = useI18n();
  const { formatMw, pick } = i18n;
  const [mode, setMode] = useState("feature");
  const reconciliation = release.reconciliation;
  const rows = mode === "feature"
    ? release.study.reliability.feature_drift.map((row) => ({ ...row, label: row.feature, value: row.psi }))
    : mode === "residual"
    ? release.study.reliability.residual_drift_monthly.map((row) => ({ ...row, label: row.month, value: row.mean_absolute_error_MW }))
    : [
        { label: pick("Before adjustment", "Uyum öncesi"), value: reconciliation.raw_mae, n: reconciliation.n },
        { label: pick("After adjustment", "Uyum sonrası"), value: reconciliation.reconciled_mae, n: reconciliation.n },
      ];
  const largestShift = mode === "feature" ? rows.reduce((current, row) => row.value > current.value ? row : current, rows[0]) : null;
  const interpretation = mode === "feature"
    ? pick(
      `${largestShift.label} changed most, with PSI ${largestShift.value.toFixed(2)} (${largestShift.status}). PSI compares each feature distribution with the reference period: below 0.10 is similar, 0.10 to below 0.25 needs watching, and 0.25 or above is a large shift.`,
      `En büyük değişim ${largestShift.label} özelliğinde: PSI ${largestShift.value.toFixed(2)} (${largestShift.status}). PSI, her özelliğin dağılımını referans dönemle karşılaştırır: 0,10 altı benzer, 0,10–0,25 arası izlenmeli, 0,25 ve üzeri büyük değişimdir.`,
    )
    : mode === "residual"
    ? rangeInterpretation(rows, pick("Monthly mean absolute error", "Aylık ortalama mutlak hata"), i18n)
    : pick(
      `Daily-total alignment changes hourly MAE from ${formatMw(reconciliation.raw_mae)} to ${formatMw(reconciliation.reconciled_mae)}. The adjusted hourly forecasts ${reconciliation.invariant.all_passed ? "sum to the daily forecast" : "do not pass the daily-total check"}.`,
      `Günlük toplam uyumu, saatlik MAE değerini ${formatMw(reconciliation.raw_mae)} seviyesinden ${formatMw(reconciliation.reconciled_mae)} seviyesine getiriyor. Uyum sonrası saatlik tahminler ${reconciliation.invariant.all_passed ? "günlük tahmin toplamıyla eşleşiyor" : "günlük toplam kontrolünü geçemiyor"}.`,
    );
  const caption = mode === "feature"
    ? `${pick("Feature distribution comparison", "Özellik dağılımı karşılaştırması")}; n = ${rows.length}.`
    : mode === "residual"
    ? `${pick("Monthly forecast error", "Aylık tahmin hatası")}; n = ${rows.reduce((sum, row) => sum + Number(row.n), 0)}.`
    : `${pick("Daily-total alignment", "Günlük toplam uyumu")}; n = ${reconciliation.n}.`;
  return (
    <div className="chapter-figure">
      <EvidenceControls controls={[{
        id: "drift-mode",
        label: pick("Reliability view", "Güvenilirlik görünümü"),
        value: mode,
        onChange: setMode,
        options: [{ value: "feature", label: pick("Feature distribution change", "Özellik dağılımı değişimi") }, { value: "residual", label: pick("Monthly forecast error", "Aylık tahmin hatası") }, { value: "reconciliation", label: pick("Daily-total alignment", "Günlük toplam uyumu") }],
      }]} />
      <LiveSelection>{pick("Showing the selected reliability view.", "Seçilen güvenilirlik görünümü gösteriliyor.")}</LiveSelection>
      <ChartFrame
        title={mode === "feature" ? pick("Feature distribution change", "Özellik dağılımı değişimi") : mode === "residual" ? pick("Monthly forecast error", "Aylık tahmin hatası") : pick("Daily-total alignment", "Günlük toplam uyumu")}
        description={`${mode} reliability evidence`}
        interpretation={interpretation}
        caption={caption}
        rows={rows}
        columns={[{ key: "label", label: mode === "feature" ? pick("Feature", "Özellik") : mode === "residual" ? pick("Berlin-local month", "Berlin yerel ayı") : pick("Output state", "Çıktı durumu") }, { key: "value", label: mode === "feature" ? "PSI" : "MAE (MW)" }, ...(mode === "feature" ? [{ key: "status", label: pick("Status", "Durum") }] : [])]}
      >
        <ResponsiveContainer width="100%" height="100%"><BarChart data={rows} accessibilityLayer margin={{ top: 16, right: 20, bottom: 20, left: 16 }}><CartesianGrid {...grid} /><XAxis dataKey="label" {...axis} /><YAxis unit={mode === "feature" ? "" : " MW"} width={72} {...axis} /><Tooltip contentStyle={tooltip} /><Legend verticalAlign="top" height={36} wrapperStyle={{ color: "var(--chart-legend)" }} /><Bar name={mode === "feature" ? pick("Population Stability Index (PSI)", "Dağılım değişim endeksi (PSI)") : "MAE (MW)"} dataKey="value" fill={mode === "reconciliation" ? "var(--chart-reconciliation)" : "var(--chart-reference)"} isAnimationActive={false} /></BarChart></ResponsiveContainer>
      </ChartFrame>
    </div>
  );
}

export function StudyViews({ release }) {
  const { pick } = useI18n();
  return (
    <>
      <section id="load-patterns" className="study-chapter" data-evidence-section aria-labelledby="load-patterns-title">
        <h2 id="load-patterns-title">{pick("Demand patterns", "Talep örüntüleri")}</h2>
        <div className="chapter-figures"><LoadProfileFigure release={release} /><TemperatureFigure release={release} /><ShapFigure release={release} /></div>
      </section>
      <section id="forecast-performance" className="study-chapter" data-evidence-section aria-labelledby="forecast-performance-title">
        <h2 id="forecast-performance-title">{pick("Forecast performance", "Tahmin performansı")}</h2>
        <div className="chapter-figures"><ForecastView release={release} embedded /><ModelComparisonFigure release={release} /><PerformanceContextFigure release={release} /></div>
      </section>
      <section id="reliability" className="study-chapter" data-evidence-section aria-labelledby="reliability-title">
        <h2 id="reliability-title">{pick("Reliability and weak points", "Güvenilirlik ve zayıf noktalar")}</h2>
        <div className="chapter-figures"><ErrorSlicesFigure release={release} /><IntervalView release={release} embedded /><DriftFigure release={release} /></div>
      </section>
      <section id="evaluation-method" className="study-chapter" data-evidence-section aria-labelledby="evaluation-method-title">
        <h2 id="evaluation-method-title">{pick("Evaluation method", "Değerlendirme yöntemi")}</h2>
        <div className="chapter-figures"><ProtocolView release={release} embedded /></div>
      </section>
    </>
  );
}
