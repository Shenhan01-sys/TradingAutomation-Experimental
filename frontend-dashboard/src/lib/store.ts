import { create } from 'zustand';

interface TradePosition {
  ticketId: string;
  asset: string;
  action: 'BUY' | 'SELL';
  lot: number;
  entryPrice: number;
  currentPnl: number;
}

interface TradingState {
  balance: number;
  equity: number;
  activePositions: TradePosition[];
  livePrices: Record<string, { bid: number; ask: number }>;
  marketStatus: Record<string, 'OPEN' | 'CLOSED'>;
  updateLivePrice: (asset: string, bid: number, ask: number) => void;
  setMarketStatus: (asset: string, status: 'OPEN' | 'CLOSED') => void;
  updatePositions: (positions: TradePosition[]) => void;
  recalculateEquity: () => void;
}

export const useTradingStore = create<TradingState>((set, get) => ({
  balance: 10000, // Initial mock balance
  equity: 10000,
  activePositions: [],
  livePrices: {
    XAUUSD: { bid: 0, ask: 0 },
    BTCUSD: { bid: 0, ask: 0 }
  },
  marketStatus: {
    XAUUSD: 'OPEN',
    BTCUSD: 'OPEN'
  },

  updateLivePrice: (asset, bid, ask) => {
    set((state) => ({
      livePrices: { ...state.livePrices, [asset]: { bid, ask } }
    }));
    get().recalculateEquity();
  },

  setMarketStatus: (asset, status) => set((state) => ({
    marketStatus: { ...state.marketStatus, [asset]: status }
  })),

  updatePositions: (positions) => {
    set({ activePositions: positions });
    get().recalculateEquity();
  },

  recalculateEquity: () => {
    const { balance, activePositions, livePrices } = get();
    // Simplified floating PNL calculation
    const floatingPnl = activePositions.reduce((total, pos) => {
      const currentPrice = pos.action === 'BUY' ? livePrices[pos.asset]?.bid : livePrices[pos.asset]?.ask;
      if (!currentPrice) return total;
      
      const diff = pos.action === 'BUY' ? currentPrice - pos.entryPrice : pos.entryPrice - currentPrice;
      // Mock calculation (would use proper contract sizes in reality)
      const pnl = diff * pos.lot * (pos.asset === 'XAUUSD' ? 100 : 1); 
      
      // Update position PNL
      pos.currentPnl = pnl;
      return total + pnl;
    }, 0);

    set({ equity: balance + floatingPnl });
  }
}));
