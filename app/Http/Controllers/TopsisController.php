<?php

namespace App\Http\Controllers;

use App\Services\TopsisService;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Throwable;

class TopsisController extends Controller
{
    /**
     * Ranks the Pareto solutions currently displayed in the table using
     * TOPSIS and returns the id of the best solution.
     *
     * Expected JSON payload (built client-side from the table + weight
     * inputs, see ParetoTableView.js):
     *   - solution_ids: array<int|string>            one id per row
     *   - values:       array<array<float>>           objective matrix, one row per solution
     *   - weights:      array<float>                  one weight per objective column
     */
    public function runTopsis(Request $request, TopsisService $topsisService): JsonResponse
    {
        $validated = $request->validate([
            'solution_ids'   => ['required', 'array', 'min:1'],
            'solution_ids.*' => ['required'],
            'values'         => ['required', 'array', 'min:1'],
            'values.*'       => ['required', 'array', 'min:1'],
            'values.*.*'     => ['required', 'numeric'],
            'weights'        => ['required', 'array', 'min:1'],
            'weights.*'      => ['required', 'numeric', 'min:0'],
        ]);

        // Every row of the objective matrix must line up 1:1 with a solution
        // id and carry exactly one value per weight.
        $weightCount = count($validated['weights']);
        $rowsAreConsistent = count($validated['solution_ids']) === count($validated['values'])
            && collect($validated['values'])->every(fn (array $row) => count($row) === $weightCount);

        if (!$rowsAreConsistent) {
            return response()->json([
                'status'  => 'error',
                'message' => 'solution_ids, values and weights must be consistently sized.',
            ], 422);
        }

        try {
            $result = $topsisService->run(
                ids: $validated['solution_ids'],
                values: $validated['values'],
                weights: array_map('floatval', $validated['weights']),
            );

            return response()->json([
                'status'           => 'success',
                'best_solution_id' => $result['best_solution_id'],
            ]);
        } catch (Throwable $e) {
            report($e);
            return response()->json([
                'status'  => 'error',
                'message' => $e->getMessage(),
            ], 500);
        }
    }
}
