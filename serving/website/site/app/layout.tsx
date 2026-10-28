import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CliqueFlowmer — Computational Alchemy",
  description: "Explore crystal representations, reconstruct structures, and optimize new material candidates with CliqueFlowmer.",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
