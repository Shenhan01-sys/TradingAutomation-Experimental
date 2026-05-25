<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\TradingController;

/*
|--------------------------------------------------------------------------
| API Routes
|--------------------------------------------------------------------------
*/

Route::prefix('v1/trading')->group(function () {
    Route::get('/settings', [TradingController::class, 'getSettings']);
    Route::post('/settings', [TradingController::class, 'updateSettings']);
    
    Route::get('/history', [TradingController::class, 'getHistory']);
    Route::get('/summary', [TradingController::class, 'getSummary']);
});
