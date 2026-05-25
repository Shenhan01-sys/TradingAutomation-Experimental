<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        // 1. broker_configs
        Schema::create('broker_configs', function (Blueprint $table) {
            $table->id();
            $table->string('broker_name');
            $table->string('api_url');
            $table->string('api_key');
            $table->timestamps();
        });

        // 2. trading_settings
        Schema::create('trading_settings', function (Blueprint $table) {
            $table->id();
            $table->decimal('max_drawdown_percent', 5, 2)->default(10.00);
            $table->decimal('risk_per_trade_percent', 5, 2)->default(1.00);
            $table->boolean('is_active')->default(true);
            $table->timestamps();
        });

        // 3. trade_history
        Schema::create('trade_history', function (Blueprint $table) {
            $table->id();
            $table->string('ticket_id')->nullable()->unique();
            $table->string('asset'); // XAUUSD / BTCUSD
            $table->string('action'); // BUY / SELL
            $table->decimal('lot_size', 8, 2);
            $table->decimal('entry_price', 15, 5);
            $table->decimal('stop_loss_price', 15, 5);
            $table->decimal('pnl', 15, 2)->nullable();
            $table->string('status'); // SUCCESS, FAILED
            $table->text('ai_reasoning')->nullable();
            $table->timestamps();
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('trade_history');
        Schema::dropIfExists('trading_settings');
        Schema::dropIfExists('broker_configs');
    }
};
