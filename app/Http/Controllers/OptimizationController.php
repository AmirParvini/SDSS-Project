<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Services\Optimization\OptimizationOrchestrator;
use App\Exceptions\OptimizationProcessException;
use App\Models\Scenario;
use Illuminate\Http\JsonResponse;
use Throwable;

class OptimizationController extends Controller
{
    public function optimize(OptimizationOrchestrator $orchestrator): JsonResponse
    {
        // $validated = $request->validate([
        //     'scenario_id' => ['required', 'integer', 'exists:scenarios,id'],
        //     ]);
        try {
            $result = $orchestrator->run(
                scenarioId: Scenario::where('active', 1)->value('id'),
                pythonPath: 'public/python/HSC_NSGA-II_SOLID/run.py',
                timeout: 1800,
            );
            return response()->json([
                'status' => 'success',
                'data'   => $result,
            ]);

        } catch (OptimizationProcessException $e) {
            // خطای شناخته‌شده‌ی دامنه — لاگ + پاسخ مشخص
            report($e); // یا Log::error با جزئیات errorOutput
            return response()->json([
                'status'  => 'error',
                'message' => 'An error occurred while executing the Python script.',
                'detail'  => $e->getMessage(),
                'errorOutput'  => $e->getErrorOutput(),
            ], 500);

        } catch (Throwable $e) {
            // هر خطای پیش‌بینی‌نشده‌ی دیگر
            report($e);
            return response()->json([
                'status'  => 'error',
                'message' => 'Internal server error! (500)',
                'detail' => $e->getMessage(),
                'errorOutput' => $e->getFile(),
                'line' => $e->getLine(),
            ], 500);
        }
    }

}
