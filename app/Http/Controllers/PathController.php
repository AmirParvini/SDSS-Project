<?php

namespace App\Http\Controllers;

use Illuminate\Http\JsonResponse;
use App\Services\Path\PathOrchestrator;
use App\Models\Scenario;
use Throwable;

class PathController extends Controller
{
    public function calculate(PathOrchestrator $path_orchestrator): JsonResponse
    {
        session_write_close(); 
        try {
            $path_orchestrator->run(
                scenarioId: Scenario::where('active', 1)->value('id'),
                pythonPath: 'public/python/PathCalculator/cli.py',
                timeout: 1800
            );
            return response()->json([
                'status' => 'success',
                'message' => 'Path calculation completed successfully.'
            ]);
        } catch (Throwable $e) {
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
