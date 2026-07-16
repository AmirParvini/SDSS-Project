<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Services\ReportService;

class ReportController extends Controller
{
    protected $reportService;

    public function __construct(ReportService $reportService)
    {
        $this->reportService = $reportService;
    }

    /**
     * Get report for a scenario.
     * Can accept scenario_id either from route parameter or query string.
     *
     * @param Request $request
     * @param int|null $scenario_id
     * @return \Illuminate\Http\JsonResponse
     */
    public function getReport(Request $request, $scenario_id = null)
    {
        // If not provided in route, look in request query/body
        $id = $scenario_id ?? $request->input('scenario_id');

        if (!$id) {
            return response()->json([
                'success' => false,
                'message' => 'Scenario ID is required.'
            ], 400);
        }

        $report = $this->reportService->generateReport((int)$id);

        if (!$report['success']) {
            return response()->json($report, 404);
        }

        return response()->json($report);
    }
}
