/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        quantum: {
          50:  '#e0f9ff',
          100: '#b3f0ff',
          200: '#66e0ff',
          300: '#1acfff',
          400: '#00bfef',
          500: '#00d4ff',   // primary quantum blue
          600: '#0099bf',
          700: '#006680',
          800: '#003340',
          900: '#001a20',
        },
        void: {
          50:  '#f0f1f5',
          100: '#d1d5e8',
          200: '#a3abd1',
          300: '#7581ba',
          400: '#4757a3',
          500: '#1a2d8c',
          600: '#0f1d6b',
          700: '#0a1347',
          800: '#060c2e',
          900: '#030616',
          950: '#010310',
        },
        tumor: '#ef4444',
        healthy: '#10b981',
        warning: '#f59e0b',
      },
      fontFamily: {
        sans:  ['Inter', 'system-ui', 'sans-serif'],
        display: ['Space Grotesk', 'Inter', 'sans-serif'],
        mono:  ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      backgroundImage: {
        'quantum-gradient': 'linear-gradient(135deg, #7c3aed 0%, #00d4ff 100%)',
        'dark-grid': `
          linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
          linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px)
        `,
      },
      boxShadow: {
        'quantum': '0 0 30px rgba(0, 212, 255, 0.15)',
        'quantum-lg': '0 0 60px rgba(0, 212, 255, 0.25)',
        'tumor': '0 0 30px rgba(239, 68, 68, 0.3)',
        'healthy': '0 0 30px rgba(16, 185, 129, 0.3)',
        'glow': '0 0 20px rgba(124, 58, 237, 0.4)',
      },
      animation: {
        'pulse-quantum': 'pulseQuantum 2s ease-in-out infinite',
        'scan':          'scan 2.5s ease-in-out infinite',
        'float':         'float 4s ease-in-out infinite',
        'glow-pulse':    'glowPulse 2s ease-in-out infinite alternate',
        'spin-slow':     'spin 8s linear infinite',
        'fade-in':       'fadeIn 0.5s ease-out',
        'slide-up':      'slideUp 0.4s ease-out',
      },
      keyframes: {
        pulseQuantum: {
          '0%, 100%': { opacity: '1', transform: 'scale(1)' },
          '50%':      { opacity: '0.7', transform: 'scale(1.05)' },
        },
        scan: {
          '0%':   { top: '0%' },
          '50%':  { top: '100%' },
          '100%': { top: '0%' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%':      { transform: 'translateY(-12px)' },
        },
        glowPulse: {
          from: { boxShadow: '0 0 10px rgba(0,212,255,0.2)' },
          to:   { boxShadow: '0 0 40px rgba(0,212,255,0.5)' },
        },
        fadeIn: {
          from: { opacity: '0' },
          to:   { opacity: '1' },
        },
        slideUp: {
          from: { opacity: '0', transform: 'translateY(20px)' },
          to:   { opacity: '1', transform: 'translateY(0)' },
        },
      },
      backdropBlur: {
        xs: '2px',
      },
    },
  },
  plugins: [],
}
