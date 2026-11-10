import './globals.css';

export const metadata = {
  title: 'Indic Shield',
  description: 'Context-aware toxicity checker for Telugu, Romanized Telugu and code-mixed chat.',
  icons: { icon: "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Crect width='16' height='16' rx='3' fill='%230f766e'/%3E%3C/svg%3E" },
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
