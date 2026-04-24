import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "var(--bg)",
        panel: "var(--panel)",
        panel2: "var(--panel-2)",
        ink: "var(--ink)",
        muted: "var(--muted)",
        line: "var(--line)",
        accent: "var(--accent)",
      },
      fontFamily: {
        sans: ["var(--font-sans)", "Inter", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "JetBrains Mono", "ui-monospace", "monospace"],
      },
      boxShadow: {
        accent: "0 0 0 1px rgba(57,255,20,0.2), 0 0 28px rgba(57,255,20,0.12)",
        panel: "0 0 0 1px rgba(38,38,38,0.9), 0 18px 60px rgba(0,0,0,0.45)",
      },
      backgroundImage: {
        grid: "linear-gradient(to right, rgba(39,255,20,0.06) 1px, transparent 1px), linear-gradient(to bottom, rgba(39,255,20,0.06) 1px, transparent 1px)",
      },
    },
  },
  plugins: [],
};

export default config;
