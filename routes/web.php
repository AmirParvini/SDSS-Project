<?php

use App\Http\Controller\Test;
use App\Http\Controllers\Api\ConfigurationController;
use App\Http\Controllers\DamagedAreaController;
use App\Http\Controllers\ECController;
use App\Http\Controllers\HomeController;
use App\Http\Controllers\HospitalController;
use App\Http\Controllers\IDCController;
use App\Http\Controllers\OptimizationController;
use App\Http\Controllers\SolvingController;
use App\Http\Controllers\TMCController;
use Illuminate\Support\Facades\Route;

Route::view('/', 'home');
Route::get("/load_data", [HomeController::class, 'index'])->name('load_data');
// Route::get("/", [OptimizationController::class, 'optimize'])->name('optima');
// Route::post('/solving',[SolvingController::class, 'index'])->name('solving.index');
// Route::get('/solving',[SolvingController::class, 'index'])->name('solving.index');
// Route::post('/api/config/getdata',[ConfigurationController::class, 'getdata'])->name('configuration.getdata');
// Route::get('/server-info', function () {
//     phpinfo();
// });
Route::resource('IDC', IDCController::class);
Route::resource('EC', ECController::class);
Route::resource('H', HospitalController::class);
Route::resource('TMC', TMCController::class);
Route::resource('DA', DamagedAreaController::class);
Route::post('/damaged-areas', [DamagedAreaController::class, 'store'])->name('damaged-areas.store');
