<?php

namespace App\Services\Path;

use App\Models\Cost;
use App\Models\HospitalAllocation;
use App\Models\PackageFlow;
use App\Models\ParetoSolution;
use App\Models\Path;
use App\Models\ShelterAllocation;
use App\Models\TemporaryMedicalCenterAllocation;

class SaveResults
{

    function store(int $scenario_id, array $results, string $path_type)
    {
        if (empty($results)) {
            return;
        }

        $rows = [];
        foreach ($results as $result) {
            $rows[] = [
                'scenario_id' => $scenario_id,
                'source_id'  => $result['source_id'],
                'target_id'  => $result['target_id'],
                'path_type'  => $path_type,
                'distance'   => $result['distance'],
                'geometry'   => json_encode($result['geometry'], JSON_UNESCAPED_UNICODE),
                'created_at' => now(),
                'updated_at' => now(),
            ];
        }

        foreach (array_chunk($rows, 500) as $chunk) {
            Path::insert($chunk);
        }
    }
}
