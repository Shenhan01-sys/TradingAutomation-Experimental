import '../app/globals.css';
import Header from '../components/Header';
import Sidebar from '../components/Sidebar';

export const metadata = {
  title: 'AI Trading Dashboard',
  description: 'Autonomous AI Trading Bot Dashboard',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-gray-950 text-gray-100 min-h-screen flex flex-col font-sans">
        <Header />
        <div className="flex flex-1 overflow-hidden">
          <Sidebar />
          <main className="flex-1 overflow-y-auto p-6">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
