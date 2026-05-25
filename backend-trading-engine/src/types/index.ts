import { FastifyBaseLogger } from 'fastify';

export type AssetSymbol = 'XAUUSD' | 'BTCUSD';
export type ActionType = 'BUY' | 'SELL' | 'CANCEL';

export interface TickData {
  symbol: AssetSymbol;
  bid: number;
  ask: number;
  timestamp: number;
}

export interface AIValidationResult {
  asset: AssetSymbol;
  action: ActionType;
  confidence_score: number;
  reasoning: string;
}

export interface TradeConfig {
  pipValue: number;
  contractSize: number;
}

export interface OrderPayload {
  symbol: AssetSymbol;
  action: ActionType;
  lotSize: number;
  price: number; // Signal execution requested price
  stopLossPrice: number;
}
