<?php

use App\Http\Controller\Test;
use App\Http\Controllers\Api\ConfigurationController;
use App\Http\Controllers\DamagedAreaController;
use App\Http\Controllers\ECController;
use App\Http\Controllers\HomeController;
use App\Http\Controllers\HospitalController;
use App\Http\Controllers\IDCController;
use App\Http\Controllers\OptimizationController;
use App\Http\Controllers\ScenarioController;
use App\Http\Controllers\SolvingController;
use App\Http\Controllers\TMCController;
use App\Http\Controllers\NodeCrudController;
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

Route::post('/nodes', [NodeCrudController::class, 'store'])->name('nodes.store');
Route::put('/nodes/{id}', [NodeCrudController::class, 'update'])->name('nodes.update');
Route::delete('/nodes/{id}', [NodeCrudController::class, 'destroy'])->name('nodes.destroy');

Route::put('/select-scenario/{current_scenario_id}', [ScenarioController::class, 'select_scenario'])->name('scenarios.select');
Route::get('/scenarios', [ScenarioController::class, 'index'])->name('scenarios.index');
Route::post('/scenarios', [ScenarioController::class, 'store'])->name('scenarios.store');
Route::put('/scenarios/{current_scenario_id}', [ScenarioController::class, 'update'])->name('scenarios.update');