<?php

namespace App\Services\Path;

use App\Exceptions\PathProcessException;
use Symfony\Component\Process\Process;

class PathRunProccess
{

    public function run(Process $process): array
    {
        $process->start();
        $process->wait();
        if (!$process->isSuccessful()) {
            throw PathProcessException::fromFailedProcess($process);
        }
        $output = json_decode($process->getOutput(), true);
        if (!is_array($output)) {
            throw PathProcessException::invalidOutput($process->getOutput());
        }
        return $output;
    }
}
