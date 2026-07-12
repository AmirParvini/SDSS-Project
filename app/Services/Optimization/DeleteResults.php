<?php

namespace App\Services\Optimization;

use App\Models\Cost;
use App\Models\HospitalAllocation;
use App\Models\PackageFlow;
use App\Models\ParetoSolution;
use App\Models\ShelterAllocation;
use App\Models\TemporaryMedicalCenterAllocation;

class DeleteResults
{
    public function delete(int $scenarioId): void
    {
        ParetoSolution::query()->where('scenario_id', $scenarioId)->delete();
        PackageFlow::query()->where('scenario_id', $scenarioId)->delete();
        TemporaryMedicalCenterAllocation::query()->where('scenario_id', $scenarioId)->delete();
        HospitalAllocation::query()->where('scenario_id', $scenarioId)->delete();
        ShelterAllocation::query()->where('scenario_id', $scenarioId)->delete();
        Cost::query()->where('scenario_id', $scenarioId)->delete();
    }
}
