<?php

namespace App\Services\Optimization;

class OptimizationOrchestrator
{
    public function __construct(
        private BuildData $buildData,
        private CreateProcess $createProcess,
        private RunProcess $runProcess,
    ) {}

    public function run(int $scenarioId, string $pythonPath, int $timeout): array
    {
        $data = $this->buildData->build($scenarioId);
        $process = $this->createProcess->create($pythonPath, $data, $timeout);
        return $this->runProcess->run($process, $scenarioId);
    }
}
