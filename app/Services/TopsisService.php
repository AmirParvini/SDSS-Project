<?php

namespace App\Services;

use Symfony\Component\Process\Process;

class TopsisService
{
    /**
     * Runs the TOPSIS MCDM algorithm (public/python/Topsis/topsis.py) against
     * the Pareto solutions currently shown on the front-end and returns the
     * id of the best-ranked solution.
     *
     * @param  array<int, int|string>       $ids      Solution ids, aligned with $values rows.
     * @param  array<int, array<int, float>> $values  Objective matrix: one row per solution (e.g. [z1, z2, z3]).
     * @param  array<int, float>            $weights  Objective weights, aligned with the $values columns.
     * @return array{best_solution_id: int|string}
     */
    public function run(array $ids, array $values, array $weights): array
    {
        $scriptPath = base_path('public/python/Topsis/topsis.py');

        $input = json_encode([
            'ids'     => array_values($ids),
            'X'       => array_values($values),
            'weights' => array_values($weights),
        ]);

        $process = new Process(['python', $scriptPath]);
        $process->setInput($input);
        $process->setEnv([
            'SystemRoot' => getenv('SystemRoot') ?: 'C:\\Windows',
            'PATH'       => getenv('PATH'),
        ]);
        $process->setTimeout(60);
        $process->run();

        if (!$process->isSuccessful()) {
            throw new \RuntimeException('Topsis script failed: '.$process->getErrorOutput());
        }

        $result = json_decode($process->getOutput(), true);

        if (!is_array($result) || !array_key_exists('best_solution_id', $result)) {
            $message = is_array($result) && isset($result['error'])
                ? $result['error']
                : 'Invalid Topsis output';

            throw new \RuntimeException($message);
        }

        return [
            'best_solution_id' => $result['best_solution_id'],
        ];
    }
}
