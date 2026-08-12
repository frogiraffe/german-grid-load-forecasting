import { createContext, useContext, useEffect, useMemo, useState } from "react";

const messages = {
  en: {
    "language.label": "Language",
    "language.en": "English",
    "language.tr": "Türkçe",
    "theme.label": "Theme",
    "theme.system": "System",
    "theme.light": "Light",
    "theme.dark": "Dark",
    "shell.skip": "Skip to evidence",
    "shell.title": "German Grid-Load Forecasting / Technical Results",
    "shell.modelCard": "Model card",
    "shell.report": "Technical report",
    "shell.repository": "Repository",
    "header.title": "Daily and hourly grid-load forecasts",
    "header.description": "Observed demand, forecast accuracy, uncertainty and reliability for daily and 24-step hourly forecasts.",
    "header.dateRange": "Date range",
    "header.paths": "Forecast paths",
    "header.pathValue": "Daily and 24-step hourly",
    "summary.label": "Results summary",
    "summary.daily": "Daily comparison",
    "summary.dailyMetric": "{model}: {mae} MAE",
    "summary.dailyBaselines": "One-day naive: {oneDay}; seven-day naive: {sevenDay}; n = {n}.",
    "summary.alignment": "Daily-total alignment",
    "summary.alignmentMetric": "{model}: {mae} adjusted MAE",
    "summary.alignmentDetail": "Before adjustment {raw}; change {change}; n = {n}.",
    "summary.interval": "Prediction interval",
    "summary.intervalMethod": "{method}, {level}",
    "summary.intervalDetail": "Observed-value coverage {coverage}; mean width {width}; score {score}; n = {n}.",
    "sections.label": "Results sections",
    "sections.demand": "Demand patterns",
    "sections.performance": "Forecast performance",
    "sections.reliability": "Reliability and weak points",
    "sections.method": "Evaluation method",
    "status.error": "The generated evidence could not be displayed. Refresh the page and try again.",
    "status.emptyTitle": "No evidence matches these filters",
    "status.emptyBody": "Choose another model, horizon, local hour, output state, or interval method.",
    "table.show": "View chart data",
    "table.values": "{title} exact values",
    "chart.interpretation": "What this shows",
    "zoom.reset": "Reset zoom",
    "footer.text": "Generated evidence for the documented evaluation protocol.",
  },
  tr: {
    "language.label": "Dil",
    "language.en": "English",
    "language.tr": "Türkçe",
    "theme.label": "Tema",
    "theme.system": "Sistem",
    "theme.light": "Açık",
    "theme.dark": "Koyu",
    "shell.skip": "Bulgulara geç",
    "shell.title": "Almanya Şebeke Yükü Tahmini / Teknik Sonuçlar",
    "shell.modelCard": "Model kartı",
    "shell.report": "Teknik rapor",
    "shell.repository": "Kod deposu",
    "header.title": "Günlük ve saatlik şebeke yükü tahminleri",
    "header.description": "Günlük ve 24 adımlı saatlik tahminler için gerçekleşen talep, tahmin doğruluğu, belirsizlik ve güvenilirlik.",
    "header.dateRange": "Tarih aralığı",
    "header.paths": "Tahmin türleri",
    "header.pathValue": "Günlük ve 24 adımlı saatlik",
    "summary.label": "Sonuç özeti",
    "summary.daily": "Günlük karşılaştırma",
    "summary.dailyMetric": "{model}: {mae} MAE",
    "summary.dailyBaselines": "Bir günlük naif: {oneDay}; yedi günlük naif: {sevenDay}; n = {n}.",
    "summary.alignment": "Günlük toplam uyumu",
    "summary.alignmentMetric": "{model}: {mae} uyum sonrası MAE",
    "summary.alignmentDetail": "Uyum öncesi {raw}; değişim {change}; n = {n}.",
    "summary.interval": "Tahmin aralığı",
    "summary.intervalMethod": "{method}, {level}",
    "summary.intervalDetail": "Gerçek değer kapsaması {coverage}; ortalama genişlik {width}; skor {score}; n = {n}.",
    "sections.label": "Sonuç bölümleri",
    "sections.demand": "Talep örüntüleri",
    "sections.performance": "Tahmin performansı",
    "sections.reliability": "Güvenilirlik ve zayıf noktalar",
    "sections.method": "Değerlendirme yöntemi",
    "status.error": "Üretilen bulgular gösterilemedi. Sayfayı yenileyip tekrar deneyin.",
    "status.emptyTitle": "Bu filtrelerle eşleşen bulgu yok",
    "status.emptyBody": "Başka bir model, ufuk, yerel saat, çıktı durumu veya aralık yöntemi seçin.",
    "table.show": "Grafik verilerini göster",
    "table.values": "{title} kesin değerleri",
    "chart.interpretation": "Bu ne gösteriyor?",
    "zoom.reset": "Yakınlaştırmayı sıfırla",
    "footer.text": "Belgelenmiş değerlendirme yöntemi için üretilen bulgular.",
  },
};

function storage() {
  try {
    return window.localStorage;
  } catch {
    return undefined;
  }
}

function interpolate(template, values = {}) {
  return template.replace(/\{(\w+)\}/g, (_, key) => String(values[key] ?? `{${key}}`));
}

function languageValue(language, setLanguage) {
  const locale = language === "tr" ? "tr-TR" : "en-GB";
  const t = (key, values) => interpolate(messages[language][key] ?? messages.en[key] ?? key, values);
  return {
    language,
    locale,
    setLanguage,
    t,
    pick(english, turkish) {
      return language === "tr" ? turkish : english;
    },
    formatDate(value, withTime = false) {
      const date = new Date(withTime ? value : `${value}T00:00:00Z`);
      return new Intl.DateTimeFormat(locale, {
        day: "2-digit",
        month: "short",
        year: "numeric",
        ...(withTime ? { hour: "2-digit", minute: "2-digit", timeZone: "Europe/Berlin" } : { timeZone: "UTC" }),
      }).format(date);
    },
    formatMw(value) {
      return `${Math.round(value).toLocaleString(locale)} MW`;
    },
    formatPercent(value) {
      return `${(Number(value) * 100).toLocaleString(locale, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}%`;
    },
    formatModel(value) {
      return String(value).replaceAll("_", " ");
    },
    formatCriterion(value) {
      return String(value).match(/(RMSE|MAPE|MASE|MAE)$/)?.[1] ?? String(value).replaceAll("_", " ");
    },
  };
}

const fallback = languageValue("en", () => {});
const LanguageContext = createContext(fallback);

export function LanguageProvider({ children }) {
  const [language, setLanguage] = useState(() =>
    storage()?.getItem("dashboard-language") ??
    (window.navigator.language.toLowerCase().startsWith("tr") ? "tr" : "en"),
  );
  useEffect(() => {
    document.documentElement.lang = language;
    storage()?.setItem("dashboard-language", language);
  }, [language]);
  const value = useMemo(() => languageValue(language, setLanguage), [language]);
  return <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>;
}

export function useI18n() {
  return useContext(LanguageContext);
}

export function LanguageControl() {
  const { language, setLanguage, t } = useI18n();
  return (
    <label className="language-control" htmlFor="language">
      {t("language.label")}
      <select id="language" value={language} onChange={(event) => setLanguage(event.target.value)}>
        <option value="en">{t("language.en")}</option>
        <option value="tr">{t("language.tr")}</option>
      </select>
    </label>
  );
}
