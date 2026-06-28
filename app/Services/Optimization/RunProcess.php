<?php

namespace App\Services\Optimization;

use Symfony\Component\Process\Process;
use App\Exceptions\OptimizationProcessException;
use Illuminate\Support\Facades\Cache;

class RunProcess
{
    public function run(Process $process, int $scenarioId): array
    {
        $process->start();
        Cache::put("optimization_pid_{$scenarioId}", $process->getPid(), now()->addMinutes(30));
        $process->wait();
        if (!$process->isSuccessful()) {
            throw OptimizationProcessException::fromFailedProcess($process);
            }
            $output = json_decode($process->getOutput(), true);
            if (!is_array($output)) {
            throw OptimizationProcessException::invalidOutput($process->getOutput());
        }
        return $output;
    }
}
