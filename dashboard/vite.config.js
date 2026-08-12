import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const testReleaseModule = "virtual:dashboard-release";

export default defineConfig({
  base: "./",
  build: {
    chunkSizeWarningLimit: 800,
    minify: false,
    modulePreload: { polyfill: false },
  },
  plugins: [
    react(),
    {
      name: "virtual-dashboard-release",
      resolveId(id) {
        if (id === testReleaseModule) return id;
      },
    },
  ],
  test: {
    alias: {
      "./generated/release.json": testReleaseModule,
    },
    environment: "jsdom",
  },
});
