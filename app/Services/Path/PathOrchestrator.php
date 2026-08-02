<?php

namespace App\Services\Path;

use App\Exceptions\PathProcessException;
use App\Models\Scenario;
use App\Services\Path\AirPathService;
use App\Services\Path\GroundPathService;
use App\Services\Path\InputPreparation;
use Throwable;

class PathOrchestrator {
    public function __construct(
        private GroundPathService $ground_path_service,
        private AirPathService $air_path_service,
        private InputPreparation $input_preparation
    ) {
    }
    function run(int $scenarioId, string $pythonPath, int $timeout) {

        $assings = $this->input_preparation->assinments($scenarioId);

        if (empty($assings["ground"]) && empty($assings["air"])) {
            return response()->json([
                'status' => 'success',
                'message' => 'No new paths to calculate.',
                'ground' => null,
                'air' => null,
            ]);
        }

        $ground_result = null;
        $air_result = null;
        $errors = [];

        // Run ground path service independently
        try {
            if (!empty($assings["ground"])) {
                $ground_result = $this->ground_path_service->run(
                    scenario_id: $scenarioId,
                    assigns: $assings["ground"],
                    pythonPath: $pythonPath,
                    timeout: $timeout,
                );
            }
        } catch (PathProcessException $e) {
            report($e);
            $errors['ground'] = [
                'message' => $e->getMessage(),
                'errorOutput' => $e->getErrorOutput()
            ];
            throw $e;
        } catch (Throwable $e) {
            report($e);
            $errors['ground'] = [
                'message' => $e->getMessage()
            ];
            throw $e;
        }

        // Run air path service independently
        try {
            if (!empty($assings["air"])) {
                $air_result = $this->air_path_service->run(
                    scenario_id: $scenarioId,
                    assigns: $assings["air"]
                );
            }
        } catch (PathProcessException $e) {
            report($e);
            $errors['air'] = [
                'message' => $e->getMessage(),
                'errorOutput' => $e->getErrorOutput()
            ];
            throw $e;
        } catch (Throwable $e) {
            report($e);
            $errors['air'] = [
                'message' => $e->getMessage()
            ];
            throw $e;
        }

        if (!empty($errors)) {
            return response()->json([
                'status' => 'partial_error',
                'ground' => $ground_result,
                'air' => $air_result,
                'errors' => $errors
            ], 207);
        }

        return response()->json([
            'status' => 'success',
            'ground'   => $ground_result,
            'air'   => $air_result,
        ]);
    }
}