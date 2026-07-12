<?php

namespace App\Services\Optimization;

class OptimizationOrchestrator
{
    public function __construct(
        private BuildData $buildData,
        private CreateProcess $createProcess,
        private RunProcess $runProcess,
        private SaveResults $save_result,
        private DeleteResults $deleteResults
    ) {}

    public function run(int $scenarioId, string $pythonPath, int $timeout): array
    {
        $data = $this->buildData->build($scenarioId);
        $process = $this->createProcess->create($pythonPath, $data, $timeout);
        $results = $this->runProcess->run($process, $scenarioId);
        $this->deleteResults->delete($scenarioId);
        $this->save_result->store($results, $scenarioId);
        return $results;
    }
}
