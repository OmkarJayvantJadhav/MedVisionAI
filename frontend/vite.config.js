import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // During development the API runs separately (`make run`).
    proxy: { "/api": "http://localhost:8000" },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.js"],
    css: false,
    coverage: { include: ["src/**/*.{js,jsx}"], exclude: ["src/main.jsx", "src/test/**"] },
  },
});
