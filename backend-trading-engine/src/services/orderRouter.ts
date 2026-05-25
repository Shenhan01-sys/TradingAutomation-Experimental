import { Queue } from 'bullmq';
import axios from 'axios';
import { FastifyBaseLogger } from 'fastify';
import Decimal from 'decimal.js';
import { OrderPayload } from '../types';

const tradeLogsQueue = new Queue('trade_logs_queue', {
  connection: {
    host: process.env.REDIS_HOST || '127.0.0.1',
    port: Number(process.env.REDIS_PORT) || 6379
  }
});

const BROKER_API_URL = process.env.BROKER_API_URL || 'http://localhost:8080/mock-broker';
const IS_PAPER_TRADING = process.env.IS_PAPER_TRADING === 'true';
const MAX_SLIPPAGE = new Decimal(process.env.MAX_SLIPPAGE || '2.0');

export async function executeOrder(
  payload: OrderPayload,
  currentMarketPrice: number,
  logger: FastifyBaseLogger
): Promise<void> {
  const signalPrice = new Decimal(payload.price);
  const executionPrice = new Decimal(currentMarketPrice);
  const slippage = signalPrice.minus(executionPrice).abs();

  if (slippage.greaterThan(MAX_SLIPPAGE)) {
    logger.warn({ symbol: payload.symbol, slippage: slippage.toNumber() }, '[OrderRouter] Max slippage exceeded. Canceling order.');
    await logTrade({ ...payload, status: 'FAILED', reason: 'MAX_SLIPPAGE_EXCEEDED' }, logger);
    return;
  }

  if (IS_PAPER_TRADING) {
    logger.info({ payload }, '[OrderRouter] [PAPER TRADING] Order executed successfully.');
    await logTrade({ ...payload, status: 'SUCCESS', ticketId: `PAPER-${Date.now()}` }, logger);
    return;
  }

  let attempts = 0;
  const maxRetries = 3;

  while (attempts < maxRetries) {
    try {
      const response = await axios.post(`${BROKER_API_URL}/order`, payload, {
        timeout: 2000
      });

      if (response.data.status === 'SUCCESS') {
        await logTrade({ ...payload, status: 'SUCCESS', ticketId: response.data.ticketId }, logger);
        return;
      }
      
      throw new Error(response.data.reason || 'Order rejected by broker');

    } catch (error: any) {
      attempts++;
      logger.warn({ err: error.message }, `[OrderRouter] Attempt ${attempts} failed for ${payload.symbol}`);
      
      if (error?.response?.data?.reason === 'INSUFFICIENT_MARGIN') {
        logger.error('[OrderRouter] Margin call. Aborting immediately.');
        await logTrade({ ...payload, status: 'FAILED', reason: 'INSUFFICIENT_MARGIN' }, logger);
        return;
      }
      
      if (attempts === maxRetries) {
        logger.error(`[OrderRouter] Max retries reached for ${payload.symbol}. Canceling order.`);
        await logTrade({ ...payload, status: 'FAILED', reason: 'MAX_RETRIES_REACHED' }, logger);
        return;
      }
      
      await new Promise(resolve => setTimeout(resolve, 500));
    }
  }
}

async function logTrade(data: any, logger: FastifyBaseLogger) {
  try {
    await tradeLogsQueue.add('log_trade', data);
  } catch (err) {
    logger.error({ err }, '[OrderRouter] Failed to push to BullMQ');
  }
}
