import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CliqueFlowmer — Optimize Crystals with AI",
  description:
    "Optimize crystals with clique and flow models. Explore material representations, reconstruct structures, and discover new candidates with CliqueFlowmer.",
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
