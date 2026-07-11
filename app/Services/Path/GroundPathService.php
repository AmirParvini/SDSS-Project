<?php

namespace App\Services\Path;

use App\Services\Optimization\CreateProcess;
use App\Services\Path\PathRunProccess;

class GroundPathService
{
    public function __construct(
        private CreateProcess $create_proccess,
        private PathRunProccess $path_run_proccess,
        private SaveResults $save_results,
    ) {}

    function run(string $pythonPath, int $timeout, array $assigns) {
        $process = $this->create_proccess->create($pythonPath, $assigns, $timeout);
        $output = $this->path_run_proccess->run($process);
        $this->save_results->store($output, "ground");
        return $output;
    }
}
