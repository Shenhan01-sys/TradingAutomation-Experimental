import Redis from 'ioredis';
import { FastifyBaseLogger } from 'fastify';
import { AssetSymbol, TickData, ActionType } from '../types';
import { validateSignalWithAI } from '../ai-agents/sentimentValidator';
import { executeOrder } from '../services/orderRouter';
import { calculateOrderSize } from '../risk-management/calculator';

const pubSub = new Redis({
  host: process.env.REDIS_HOST || '127.0.0.1',
  port: Number(process.env.REDIS_PORT) || 6379
});

const ACCOUNT_BALANCE = 10000;
const RISK_PERCENT = 1; 

export async function initializeStrategyEngine(logger: FastifyBaseLogger) {
  await pubSub.subscribe('market_ticks');

  pubSub.on('message', async (channel, message) => {
    if (channel === 'market_ticks') {
      try {
        const tick: TickData = JSON.parse(message);
        await processTick(tick, logger);
      } catch (err) {
        logger.error({ err }, '[StrategyEngine] Error processing tick');
      }
    }
  });
  
  logger.info('Strategy Engine initialized and listening for ticks.');
}

async function processTick(tick: TickData, logger: FastifyBaseLogger) {
  const technicalSignal = calculateTechnicalSignal(tick);
  
  if (technicalSignal === 'CANCEL') return;

  logger.info(`[StrategyEngine] Technical Signal Triggered: ${technicalSignal} on ${tick.symbol}`);

  const validation = await validateSignalWithAI(tick.symbol, technicalSignal, logger);
  
  if (validation.action === 'CANCEL' || validation.confidence_score < 0.7) {
    logger.info(`[StrategyEngine] Trade Cancelled by AI. Confidence: ${validation.confidence_score}. Reason: ${validation.reasoning}`);
    return;
  }

  const stopLossPips = 50; 
  const lotSize = calculateOrderSize(ACCOUNT_BALANCE, RISK_PERCENT, stopLossPips, tick.symbol, logger);

  if (lotSize <= 0) {
    logger.warn(`[StrategyEngine] Invalid lot size calculated. Trade aborted.`);
    return;
  }

  const signalPrice = technicalSignal === 'BUY' ? tick.ask : tick.bid;
  
  // Example mock current market price slightly different to simulate slippage
  const currentMarketPrice = signalPrice + 0.5;

  await executeOrder({
    symbol: tick.symbol,
    action: technicalSignal,
    lotSize,
    price: signalPrice,
    stopLossPrice: technicalSignal === 'BUY' ? signalPrice - (stopLossPips * 0.01) : signalPrice + (stopLossPips * 0.01)
  }, currentMarketPrice, logger);
}

function calculateTechnicalSignal(tick: TickData): ActionType {
  // Placeholder logic
  return 'CANCEL';
}
