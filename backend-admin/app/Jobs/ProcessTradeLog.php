<?php

namespace App\Jobs;

use App\Models\TradeHistory;
use Illuminate\Bus\Queueable;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Foundation\Bus\Dispatchable;
use Illuminate\Queue\InteractsWithQueue;
use Illuminate\Queue\SerializesModels;
use Illuminate\Support\Facades\Log;

class ProcessTradeLog implements ShouldQueue
{
    use Dispatchable, InteractsWithQueue, Queueable, SerializesModels;

    protected $tradeData;

    /**
     * Create a new job instance.
     */
    public function __construct(array $tradeData)
    {
        $this->tradeData = $tradeData;
    }

    /**
     * Execute the job.
     * This will read from the Redis Queue managed by BullMQ.
     */
    public function handle(): void
    {
        try {
            TradeHistory::create([
                'ticket_id' => $this->tradeData['ticketId'] ?? null,
                'asset' => $this->tradeData['symbol'],
                'action' => $this->tradeData['action'],
                'lot_size' => $this->tradeData['lotSize'],
                'entry_price' => $this->tradeData['price'],
                'stop_loss_price' => $this->tradeData['stopLossPrice'],
                'status' => $this->tradeData['status'],
                // Optional attributes based on success or failure reason
                'ai_reasoning' => $this->tradeData['reason'] ?? null,
            ]);

            Log::info("Trade Log processed for Ticket: " . ($this->tradeData['ticketId'] ?? 'Failed Order'));
        } catch (\Exception $e) {
            Log::error("Failed to process trade log: " . $e->getMessage());
            // Retry mechanisms can be handled by Laravel's queue manager
            $this->release(60); 
        }
    }
}
