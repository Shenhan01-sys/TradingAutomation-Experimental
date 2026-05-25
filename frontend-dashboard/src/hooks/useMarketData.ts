import { useEffect, useRef } from 'react';
import { useTradingStore } from '../lib/store';

export function useMarketData() {
  const wsRef = useRef<WebSocket | null>(null);
  const updateLivePrice = useTradingStore((state) => state.updateLivePrice);

  useEffect(() => {
    // Connect to Fastify WebSocket Backend
    const ws = new WebSocket('ws://localhost:3000/ws/market-data');
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('Connected to Market Data Stream');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.symbol && data.bid && data.ask) {
          updateLivePrice(data.symbol, data.bid, data.ask);
        }
      } catch (err) {
        console.error('Failed to parse WS message', err);
      }
    };

    ws.onclose = () => {
      console.log('Disconnected from Market Data Stream');
      // Reconnection logic could go here
    };

    return () => {
      ws.close();
    };
  }, [updateLivePrice]);
}
