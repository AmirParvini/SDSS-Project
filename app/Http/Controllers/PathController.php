<?php

namespace App\Http\Controllers;

use App\Models\Scenario;
use App\Services\Path\PathOrchestrator;

class PathController extends Controller
{
    function pathCalculator(PathOrchestrator $path_orchestrator)
    {
        $result = $path_orchestrator->run(
            scenarioId: Scenario::where('active', 1)->value('id'),
            pythonPath: 'public/python/PathCalculator/cli.py',
            timeout: 1800
        );
        return $result;
    }
}
