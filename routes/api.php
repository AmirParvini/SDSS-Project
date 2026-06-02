<?php

use App\Http\Controllers\GetParametersController;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;

Route::get('/user', function (Request $request) {
    return $request->user();
})->middleware('auth:sanctum');


Route::resource('/v1/getparam', GetParametersController::class);
