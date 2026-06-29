<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Services\Optimization\OptimizationOrchestrator;
use App\Exceptions\OptimizationProcessException;
use Illuminate\Http\JsonResponse;
use Throwable;

class OptimizationController extends Controller
{
    public function optimize(Request $request, OptimizationOrchestrator $orchestrator): JsonResponse
    {
        // $validated = $request->validate([
        //     'scenario_id' => ['required', 'integer', 'exists:scenarios,id'],
        //     ]);
        try {
            $result = $orchestrator->run(
                scenarioId: $request['scenario_id'],
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
                'detail' => $e->getMessage()
            ], 500);
        }
    }

}
