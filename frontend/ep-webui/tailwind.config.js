/** @type {import('tailwindcss').Config} */
export default {
    content: [
      "./pages/**/*.{js,ts,jsx,tsx,mdx}",
      "./components/**/*.{js,ts,jsx,tsx,mdx}",
      "./app/**/*.{js,ts,jsx,tsx,mdx}",
    ],
    theme: {
      extend: {
        fontFamily: {
            sans: ['InterVariable', 'system-ui', 'sans-serif'],
        },
      },
    },
    plugins: [],
  };
  