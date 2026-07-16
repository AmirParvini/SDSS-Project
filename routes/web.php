<?php

use App\Http\Controller\Test;
use App\Http\Controllers\Api\ConfigurationController;
use App\Http\Controllers\DamagedAreaController;
use App\Http\Controllers\ECController;
use App\Http\Controllers\HomeController;
use App\Http\Controllers\HospitalController;
use App\Http\Controllers\HscParametersController;
use App\Http\Controllers\IDCController;
use App\Http\Controllers\OptimizationController;
use App\Http\Controllers\ScenarioController;
use App\Http\Controllers\SolvingController;
use App\Http\Controllers\TMCController;
use App\Http\Controllers\NodeCrudController;
use App\Http\Controllers\PathController;
use App\Http\Controllers\ReportController;
use Illuminate\Support\Facades\Route;

Route::view('/', 'home');
Route::get("/load_data", [HomeController::class, 'index'])->name('load_data');
Route::post("/optima", [OptimizationController::class, 'optimize'])->name('optima');
Route::post("/path", [PathController::class, 'pathCalculator'])->name('path');
Route::get('/scenario-report/{scenario_id?}', [ReportController::class, 'getReport'])->name('scenario.report');
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

Route::get('/hsc_parameters/{scenario_id}', [HscParametersController::class, 'edit'])->name('HscParams.edit');
Route::put('/hsc_parameters/{scenario_id}', [HscParametersController::class, 'update'])->name('HscParams.update');
Route::post('/hsc_parameters', [HscParametersController::class, 'store'])->name('HscParams.store');