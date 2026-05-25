<?php

namespace App\Http\Controllers;

use App\Models\TradeHistory;
use App\Models\TradingSetting;
use Illuminate\Http\Request;
use Illuminate\Http\JsonResponse;

class TradingController extends Controller
{
    /**
     * Retrieve global trading settings.
     */
    public function getSettings(): JsonResponse
    {
        $settings = TradingSetting::first();
        return response()->json(['data' => $settings]);
    }

    /**
     * Update trading settings safely.
     */
    public function updateSettings(Request $request): JsonResponse
    {
        $validated = $request->validate([
            'max_drawdown_percent' => 'sometimes|numeric|min:0.1|max:50',
            'risk_per_trade_percent' => 'sometimes|numeric|min:0.1|max:10',
            'is_active' => 'sometimes|boolean'
        ]);

        $setting = TradingSetting::firstOrCreate([]);
        $setting->update($validated);

        return response()->json(['message' => 'Settings updated successfully', 'data' => $setting]);
    }

    /**
     * Fetch all trade history with pagination.
     */
    public function getHistory(Request $request): JsonResponse
    {
        $history = TradeHistory::orderBy('created_at', 'desc')->paginate(20);
        return response()->json($history);
    }

    /**
     * Calculate summary metrics like Win Rate, Total PnL.
     */
    public function getSummary(): JsonResponse
    {
        $totalTrades = TradeHistory::where('status', 'SUCCESS')->count();
        $winningTrades = TradeHistory::where('status', 'SUCCESS')->where('pnl', '>', 0)->count();
        
        $winRate = $totalTrades > 0 ? round(($winningTrades / $totalTrades) * 100, 2) : 0;
        $totalPnL = TradeHistory::where('status', 'SUCCESS')->sum('pnl');

        return response()->json([
            'total_trades' => $totalTrades,
            'win_rate_percent' => $winRate,
            'total_pnl' => $totalPnL
        ]);
    }
}
