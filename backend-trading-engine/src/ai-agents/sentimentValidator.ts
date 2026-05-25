import OpenAI from 'openai';
import { FastifyBaseLogger } from 'fastify';
import { AssetSymbol, ActionType, AIValidationResult } from '../types';

const openai = new OpenAI({
  baseURL: 'https://openrouter.ai/api/v1',
  apiKey: process.env.OPENROUTER_API_KEY || 'dummy_key',
});

export async function validateSignalWithAI(
  asset: AssetSymbol, 
  technicalAction: ActionType,
  logger: FastifyBaseLogger
): Promise<AIValidationResult> {
  const isXAU = asset === 'XAUUSD';
  
  const systemPrompt = isXAU 
    ? "You are an elite macro-economic AI analyst. Focus on The Fed, NFP, CPI data, USD index, and Geopolitical events. Determine if current conditions support a technical signal."
    : "You are an elite crypto quantitative AI. Focus on on-chain sentiment, ETF inflow/outflow, liquidation data, and major crypto news. Determine if current conditions support a technical signal.";
    
  const userPrompt = `Technical signal triggered: ${technicalAction} for ${asset}. Analyze current global sentiment and provide validation.`;

  try {
    const response = await openai.chat.completions.create({
      model: 'openai/gpt-4o-mini',
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt }
      ],
      response_format: { type: "json_object" },
      timeout: 5000, 
    });

    const content = response.choices[0].message?.content;
    if (!content) throw new Error("Empty AI response");

    const parsed: AIValidationResult = JSON.parse(content);
    return parsed;
  } catch (error) {
    logger.error({ err: error, asset }, `[AI Validator] Failed to validate ${asset}. Fallback to CANCEL.`);
    return {
      asset,
      action: 'CANCEL',
      confidence_score: 0,
      reasoning: 'AI validation timeout or failure. Fail-safe triggered.'
    };
  }
}
