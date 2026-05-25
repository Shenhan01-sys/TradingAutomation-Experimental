import { FastifyInstance } from 'fastify';
import Redis from 'ioredis';
import { TickData } from '../types';
import { Type } from '@sinclair/typebox';

const redis = new Redis({
  host: process.env.REDIS_HOST || '127.0.0.1',
  port: Number(process.env.REDIS_PORT) || 6379
});

export default async function brokerWs(fastify: FastifyInstance) {
  // Using TypeBox for basic endpoint validation if we had REST, 
  // but for WS we validate incoming JSON payload manually or with ajv
  fastify.get('/ws/market-data', { websocket: true }, (connection, req) => {
    fastify.log.info('Broker WebSocket Connected');

    connection.socket.on('message', async (message: string) => {
      try {
        const data: TickData = JSON.parse(message);
        
        const isXauClosed = checkXauMarketClosed(new Date(data.timestamp));
        if (data.symbol === 'XAUUSD' && isXauClosed) {
          return; 
        }

        await redis.hset(`live_price:${data.symbol}`, {
          bid: data.bid,
          ask: data.ask,
          timestamp: data.timestamp
        });

        await redis.publish('market_ticks', JSON.stringify(data));
      } catch (error) {
        fastify.log.error({ err: error }, 'Error processing websocket message');
      }
    });

    connection.socket.on('close', () => {
      fastify.log.info('Broker WebSocket Disconnected');
    });
  });
}

function checkXauMarketClosed(date: Date): boolean {
  const day = date.getUTCDay();
  const hour = date.getUTCHours();
  
  if (day === 5 && hour >= 22) return true;
  if (day === 6) return true; 
  if (day === 0 && hour < 22) return true;
  
  return false;
}
