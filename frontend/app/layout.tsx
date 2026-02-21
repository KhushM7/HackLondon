import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'OrbitGuard',
  description: 'Satellite conjunction screening and visualisation'
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <main className="shell">{children}</main>
      </body>
    </html>
  );
}
