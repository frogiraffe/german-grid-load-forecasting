import { useState } from "react";
import { Brush } from "recharts";

import { useI18n } from "../i18n.jsx";

export function formatDate(value, withTime = false) {
  const date = new Date(withTime ? value : `${value}T00:00:00Z`);
  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    ...(withTime ? { hour: "2-digit", minute: "2-digit", timeZone: "Europe/Berlin" } : { timeZone: "UTC" }),
  }).format(date);
}

export function formatMw(value) {
  return `${Math.round(value).toLocaleString("en-GB")} MW`;
}

export function formatPercent(value) {
  return `${(Number(value) * 100).toFixed(2)}%`;
}

export function formatModel(value) {
  return String(value).replaceAll("_", " ");
}

export function formatCriterion(value) {
  return String(value).match(/(RMSE|MAPE|MASE|MAE)$/)?.[1] ?? formatModel(value);
}

export function MetricSummary({ title, accent, children }) {
  return <article className={`metric-summary metric-summary--${accent}`} data-summary><h2>{title}</h2>{children}</article>;
}

export function EvidenceControls({ controls }) {
  return (
    <div className="evidence-controls">
      {controls.map(({ id, label, value, onChange, options }) => (
        <label key={id} htmlFor={id}>
          {label}
          <select id={id} value={value} onChange={(event) => onChange(event.target.value)}>
            {options.map((option) => {
              const item = typeof option === "object" ? option : { value: option, label: option };
              return <option key={String(item.value)} value={item.value}>{item.label}</option>;
            })}
          </select>
        </label>
      ))}
    </div>
  );
}

export function StatusMessage({ type = "empty" }) {
  const { t } = useI18n();
  if (type === "error") {
    return <div className="status-message status-message--error" role="alert">{t("status.error")}</div>;
  }
  return (
    <div className="status-message">
      <h3>{t("status.emptyTitle")}</h3>
      <p>{t("status.emptyBody")}</p>
    </div>
  );
}

export function EvidenceTable({ caption, columns, rows }) {
  const { t } = useI18n();
  return (
    <details>
      <summary>{t("table.show")}</summary>
      <div className="table-scroll" tabIndex="0">
        <table>
          <caption>{caption}</caption>
          <thead><tr>{columns.map((column) => <th key={column.key} scope="col">{column.label}</th>)}</tr></thead>
          <tbody>
            {rows.map((row, index) => (
              <tr key={row.id ?? `${caption}-${index}`}>
                {columns.map((column, columnIndex) => {
                  const Cell = columnIndex === 0 ? "th" : "td";
                  return <Cell key={column.key} {...(columnIndex === 0 ? { scope: "row" } : {})}>{column.render ? column.render(row[column.key], row) : row[column.key]}</Cell>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

export function ZoomBrush({ rows, dataKey, version }) {
  if (rows.length <= 12) return null;
  return <Brush key={version} dataKey={dataKey} height={24} stroke="var(--chart-forecast)" travellerWidth={10} />;
}

export function ChartFrame({ title, headingLevel = 3, description, interpretation, caption, columns, rows, zoomDataKey, children }) {
  const { t } = useI18n();
  const [zoomVersion, setZoomVersion] = useState(0);
  if (!rows.length) return <div data-chart-frame><StatusMessage /></div>;
  const Heading = `h${headingLevel}`;
  const zoomable = rows.length > 12 && Boolean(zoomDataKey);
  return (
    <figure data-chart-frame>
      <Heading>{title}</Heading>
      <div className="figure-layout">
        <div className={`chart${zoomable ? " chart--zoomable" : ""}`} role="img" aria-label={description}>
          {typeof children === "function" ? children({ zoomVersion }) : children}
        </div>
        <aside className="interpretation" data-interpretation aria-label={t("chart.interpretation")}>
          <strong>{t("chart.interpretation")}</strong>
          <p>{interpretation}</p>
        </aside>
      </div>
      {zoomable && <div className="zoom-control" data-chart-brush><button type="button" onClick={() => setZoomVersion((value) => value + 1)}>{t("zoom.reset")}</button></div>}
      <figcaption>{caption}</figcaption>
      <EvidenceTable caption={t("table.values", { title })} columns={columns} rows={rows} />
    </figure>
  );
}

export function LiveSelection({ children }) {
  return <p className="selection-status" role="status" aria-live="polite">{children}</p>;
}
