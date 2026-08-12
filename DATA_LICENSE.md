# Data licenses

The source code uses the MIT license. The data files retain their provider
licenses.

## SMARD

- Provider: Bundesnetzagentur
- Source: [SMARD.de](https://www.smard.de/en)
- Series: realized electricity consumption and grid load for Germany
- License: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
- Attribution: `Bundesnetzagentur | SMARD.de`

The pipeline converts hourly values to Europe/Berlin time. It calculates daily
means and joins the values with weather data.

## Open-Meteo

- Provider: Open-Meteo
- Historical source:
  [Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- Forecast source:
  [Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api)
- Variables: 2 m temperature and 10 m wind speed
- License: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
- Attribution: `Weather data by Open-Meteo.com`

The pipeline aggregates hourly city data to daily means. It combines city data
with fixed population weights.

Users must keep the provider attribution when they redistribute modified data.
Users must also identify their modifications.
