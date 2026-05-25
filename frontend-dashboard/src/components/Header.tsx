import { useTradingStore } from '../lib/store';

export default function Header() {
  const { balance, equity, marketStatus, livePrices } = useTradingStore();

  return (
    <header className="flex items-center justify-between px-6 py-4 bg-gray-900 border-b border-gray-800 text-white">
      <div className="flex items-center space-x-6">
        <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
          AI TradeBot
        </h1>
        
        <div className="flex space-x-4">
          <div className="flex items-center space-x-2">
            <span className="text-sm text-gray-400">XAUUSD</span>
            <span className={`h-2 w-2 rounded-full ${marketStatus.XAUUSD === 'OPEN' ? 'bg-green-500' : 'bg-red-500'}`}></span>
            <span className="text-sm font-mono">{livePrices.XAUUSD.ask || '---'}</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-sm text-gray-400">BTCUSD</span>
            <span className={`h-2 w-2 rounded-full ${marketStatus.BTCUSD === 'OPEN' ? 'bg-green-500' : 'bg-red-500'}`}></span>
            <span className="text-sm font-mono">{livePrices.BTCUSD.ask || '---'}</span>
          </div>
        </div>
      </div>

      <div className="flex space-x-6 font-mono">
        <div className="flex flex-col items-end">
          <span className="text-xs text-gray-400">Balance</span>
          <span className="text-lg font-semibold">${balance.toFixed(2)}</span>
        </div>
        <div className="flex flex-col items-end">
          <span className="text-xs text-gray-400">Equity</span>
          <span className={`text-lg font-semibold ${equity >= balance ? 'text-green-400' : 'text-red-400'}`}>
            ${equity.toFixed(2)}
          </span>
        </div>
      </div>
    </header>
  );
}
