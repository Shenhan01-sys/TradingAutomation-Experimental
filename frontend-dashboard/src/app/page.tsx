'use client';

import { useEffect } from 'react';
import { useMarketData } from '../hooks/useMarketData';
import { useTradingStore } from '../lib/store';

export default function Dashboard() {
  useMarketData(); // Initialize WebSocket connection
  const activePositions = useTradingStore(state => state.activePositions);

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Live Trading Dashboard</h2>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gray-900 border border-gray-800 p-6 rounded-xl shadow-lg">
          <h3 className="text-gray-400 text-sm font-medium">Win Rate</h3>
          <p className="text-3xl font-bold text-white mt-2">--%</p>
        </div>
        <div className="bg-gray-900 border border-gray-800 p-6 rounded-xl shadow-lg">
          <h3 className="text-gray-400 text-sm font-medium">Total AI Signals</h3>
          <p className="text-3xl font-bold text-white mt-2">--</p>
        </div>
        <div className="bg-gray-900 border border-gray-800 p-6 rounded-xl shadow-lg">
          <h3 className="text-gray-400 text-sm font-medium">Max Drawdown</h3>
          <p className="text-3xl font-bold text-red-400 mt-2">--%</p>
        </div>
      </div>

      {/* Data Tables */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Active Trades */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden shadow-lg">
          <div className="px-6 py-4 border-b border-gray-800">
            <h3 className="font-semibold text-white">Active Positions</h3>
          </div>
          <div className="p-6">
            {activePositions.length === 0 ? (
              <p className="text-gray-500 text-sm">No active positions currently.</p>
            ) : (
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="text-gray-500">
                    <th className="pb-3">Asset</th>
                    <th className="pb-3">Action</th>
                    <th className="pb-3">Entry</th>
                    <th className="pb-3 text-right">PnL</th>
                  </tr>
                </thead>
                <tbody>
                  {activePositions.map((pos) => (
                    <tr key={pos.ticketId} className="border-t border-gray-800">
                      <td className="py-3 font-medium">{pos.asset}</td>
                      <td className={`py-3 ${pos.action === 'BUY' ? 'text-green-400' : 'text-red-400'}`}>{pos.action}</td>
                      <td className="py-3 font-mono">{pos.entryPrice}</td>
                      <td className={`py-3 text-right font-mono ${pos.currentPnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        ${pos.currentPnl.toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>

        {/* AI Sentiment Logs */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden shadow-lg">
          <div className="px-6 py-4 border-b border-gray-800">
            <h3 className="font-semibold text-white">Latest AI Sentiments</h3>
          </div>
          <div className="p-6 space-y-4 text-sm">
             <div className="p-3 bg-gray-800/50 rounded-lg border border-gray-700/50">
                <div className="flex justify-between items-center mb-2">
                  <span className="font-medium text-blue-400">BTCUSD</span>
                  <span className="text-xs text-gray-500">Just now</span>
                </div>
                <p className="text-gray-300">Strong ETF inflows detected. On-chain sentiment bullish. Technical signal VALIDATED.</p>
             </div>
             <div className="p-3 bg-gray-800/50 rounded-lg border border-gray-700/50">
                <div className="flex justify-between items-center mb-2">
                  <span className="font-medium text-yellow-400">XAUUSD</span>
                  <span className="text-xs text-gray-500">2h ago</span>
                </div>
                <p className="text-gray-300">DXY strengthening ahead of NFP. Technical BUY signal REJECTED due to macro headwinds.</p>
             </div>
          </div>
        </div>
      </div>
    </div>
  );
}
