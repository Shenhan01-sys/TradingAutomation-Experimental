import Fastify from 'fastify';
import { TypeBoxTypeProvider } from '@fastify/type-provider-typebox';
import websocketPlugin from '@fastify/websocket';
import dotenv from 'dotenv';
import brokerWs from './connectors/brokerWs';
import { initializeStrategyEngine } from './strategies/mainEngine';

dotenv.config();

const fastify = Fastify({
  logger: {
    transport: {
      target: 'pino-pretty',
      options: {
        translateTime: 'SYS:standard',
        ignore: 'pid,hostname',
      },
    },
  },
}).withTypeProvider<TypeBoxTypeProvider>();

async function startServer() {
  try {
    await fastify.register(websocketPlugin);
    await fastify.register(brokerWs);

    // Pass the Fastify logger to the strategy engine
    await initializeStrategyEngine(fastify.log);

    await fastify.listen({ 
      port: Number(process.env.PORT) || 3000, 
      host: process.env.HOST || '0.0.0.0' 
    });
    
    fastify.log.info('🚀 Autonomous Trading Engine is running');
    if (process.env.IS_PAPER_TRADING === 'true') {
      fastify.log.info('⚠️ RUNNING IN PAPER TRADING MODE');
    }
  } catch (err) {
    fastify.log.error(err);
    process.exit(1);
  }
}

process.on('SIGINT', async () => {
  fastify.log.info('Shutting down gracefully...');
  await fastify.close();
  process.exit(0);
});

startServer();
