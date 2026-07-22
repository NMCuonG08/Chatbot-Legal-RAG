/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Theme-aware tokens via CSS variables (see index.css :root / .dark)
        ink: 'rgb(var(--c-ink) / <alpha-value>)',
        paper: 'rgb(var(--c-paper) / <alpha-value>)',
        'paper-dim': 'rgb(var(--c-paper-dim) / <alpha-value>)',
        'paper-tint': 'rgb(var(--c-paper-tint) / <alpha-value>)',
        rule: 'rgb(var(--c-rule) / <alpha-value>)',
        muted: 'rgb(var(--c-muted) / <alpha-value>)',
        faint: 'rgb(var(--c-faint) / <alpha-value>)',
        // VN red accent — same in both themes
        vn: {
          50: '#FFF1F3',
          100: '#FFE0E5',
          400: '#E84A60',
          500: '#D7263D',
          600: '#B91C3A',
          700: '#9B1233',
          900: '#4A0815',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
        serif: ['Source Serif 4', 'Georgia', 'serif'],
      },
      borderRadius: {
        'swiss': '2px',
      },
      letterSpacing: {
        'label': '0.16em',
        'display': '-0.02em',
      },
      animation: {
        'fade-in': 'fadeIn 0.25s ease-out forwards',
        'slide-up': 'slideUp 0.3s ease-out forwards',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: 0 },
          '100%': { opacity: 1 },
        },
        slideUp: {
          '0%': { transform: 'translateY(8px)', opacity: 0 },
          '100%': { transform: 'translateY(0)', opacity: 1 },
        },
      },
    },
  },
  plugins: [],
}