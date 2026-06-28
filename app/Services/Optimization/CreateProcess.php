<?php

namespace App\Services\Optimization;

use Symfony\Component\Process\Process;

class CreateProcess
{
    public function create(string $pythonPath, array $data, int $timeout): Process
    {
        $pythonScriptPath = base_path($pythonPath);
        $json_data = json_encode($data);
        $process = new Process(['python', $pythonScriptPath]);
        $process->setInput($json_data);
        $process->setTimeout($timeout);
        return $process;
    }
}
