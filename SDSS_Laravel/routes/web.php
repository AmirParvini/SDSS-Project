<?php

use App\Http\Controllers\Api\ConfigurationController;
use App\Http\Controllers\SolvingController;
use Illuminate\Support\Facades\Route;

Route::get('/', function () {
    return view('solving');
});

Route::get('/solving',[SolvingController::class, 'index'])->name('solving.index');
Route::post('/api/config/getdata',[ConfigurationController::class, 'getdata'])->name('configuration.getdata');

