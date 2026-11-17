import "./globals.css";

export const metadata = {
  title: "Context-Aware Toxicity Engine | Multilingual Research Console",
  description: "Hierarchical Multilingual Architecture · Target-Bypass Classification Head · Conservative Expert Routing",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
