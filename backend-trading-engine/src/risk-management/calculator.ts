import Decimal from 'decimal.js';
import { FastifyBaseLogger } from 'fastify';
import { AssetSymbol, TradeConfig } from '../types';

const ASSET_CONFIGS: Record<AssetSymbol, TradeConfig> = {
  XAUUSD: { pipValue: 10, contractSize: 100 },
  BTCUSD: { pipValue: 1, contractSize: 1 }
};

export function calculateOrderSize(
  balance: number,
  riskPercent: number,
  stopLossPips: number,
  asset: AssetSymbol,
  logger: FastifyBaseLogger
): number {
  try {
    const config = ASSET_CONFIGS[asset];
    
    const decimalBalance = new Decimal(balance);
    const riskAmount = decimalBalance.mul(new Decimal(riskPercent).div(100));
    const riskPerLot = new Decimal(stopLossPips).mul(config.pipValue);
    
    if (riskPerLot.isZero()) return 0;

    const lotSize = riskAmount.div(riskPerLot);
    
    return lotSize.toDecimalPlaces(2, Decimal.ROUND_DOWN).toNumber();
  } catch (error) {
    logger.error({ err: error, asset }, `[RiskCalculator] Error calculating order size for ${asset}`);
    return 0; // Fail-safe
  }
}
