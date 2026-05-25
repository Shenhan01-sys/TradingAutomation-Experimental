export default function Sidebar() {
  return (
    <aside className="w-64 bg-gray-900 border-r border-gray-800 text-gray-400 hidden md:flex flex-col">
      <nav className="flex-1 py-6 px-4 space-y-2">
        <a href="#" className="flex items-center space-x-3 px-3 py-2 bg-gray-800 text-white rounded-lg">
          <span>Dashboard</span>
        </a>
        <a href="#" className="flex items-center space-x-3 px-3 py-2 hover:bg-gray-800 hover:text-white rounded-lg transition">
          <span>Trade History</span>
        </a>
        <a href="#" className="flex items-center space-x-3 px-3 py-2 hover:bg-gray-800 hover:text-white rounded-lg transition">
          <span>Settings</span>
        </a>
      </nav>
      <div className="p-4 border-t border-gray-800 text-xs">
        System Status: <span className="text-green-400">Online</span>
      </div>
    </aside>
  );
}
