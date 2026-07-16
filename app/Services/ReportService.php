<?php

namespace App\Services;

use App\Models\ShelterAllocation;
use App\Models\PackageFlow;
use App\Models\HospitalAllocation;
use App\Models\TemporaryMedicalCenterAllocation;
use App\Models\Cost;
use App\Models\Scenario;
use App\Models\ParetoSolution;
use App\Models\ShelterShortage;
use App\Models\HospitalShortage;
use App\Models\TmcShortage;
use App\Models\Path;

class ReportService
{
    /**
     * Get a comprehensive report of scenario outcomes formatted exactly like
     * the _decode_one method in decoder.py.
     *
     * @param int $scenarioId
     * @return array
     */
    public function generateReport(int $scenarioId): array
    {
        ini_set('memory_limit', '512M');
        
        $scenario = Scenario::find($scenarioId);

        if (!$scenario) {
            return [
                'success' => false,
                'message' => 'Scenario not found'
            ];
        }

        // Bulk load all relevant models for the given scenario to prevent N+1 query issues (highly optimal)
        $solutions = ParetoSolution::where('scenario_id', $scenarioId)->get();
        
        $shelterAllocationsGrouped = ShelterAllocation::where('scenario_id', $scenarioId)->get()->groupBy('solution_id');
        $packageFlowsGrouped = PackageFlow::where('scenario_id', $scenarioId)->get()->groupBy('solution_id');
        $hospitalAllocationsGrouped = HospitalAllocation::where('scenario_id', $scenarioId)->get()->groupBy('solution_id');
        $tmcAllocationsGrouped = TemporaryMedicalCenterAllocation::where('scenario_id', $scenarioId)->get()->groupBy('solution_id');
        $costsGrouped = Cost::where('scenario_id', $scenarioId)->get()->groupBy('solution_id');

        $shelterShortagesGrouped = ShelterShortage::where('scenario_id', $scenarioId)->get()->groupBy('solution_id');
        $hospitalShortagesGrouped = HospitalShortage::where('scenario_id', $scenarioId)->get()->groupBy('solution_id');
        $tmcShortagesGrouped = TmcShortage::where('scenario_id', $scenarioId)->get()->groupBy('solution_id');

        // Load and group paths
        $paths = Path::where('scenario_id', $scenarioId)->get()->groupBy(function($path) {
            return "{$path->source_id},{$path->target_id},{$path->path_type}";
        });

        // Map helper to decode path geometry efficiently
        $getGeometry = function($sourceId, $targetId, $pathType) use ($paths) {
            $path = $paths->get("{$sourceId},{$targetId},{$pathType}")?->first();
            if ($path && $path->geometry) {
                $decoded = json_decode($path->geometry, true);
                return is_array($decoded) ? $decoded : $path->geometry;
            }
            return null;
        };

        $decodedSolutions = [];

        foreach ($solutions as $sol) {
            $solId = $sol->solution_id;

            // Get allocations and costs for this solution
            $sAllocations = $shelterAllocationsGrouped->get($solId, collect());
            $pFlows = $packageFlowsGrouped->get($solId, collect());
            $hAllocations = $hospitalAllocationsGrouped->get($solId, collect());
            $tAllocations = $tmcAllocationsGrouped->get($solId, collect());
            $costRow = $costsGrouped->get($solId, collect())->first();

            // Get shortages for this solution
            $sShortages = $shelterShortagesGrouped->get($solId, collect());
            $hShortages = $hospitalShortagesGrouped->get($solId, collect());
            $tShortages = $tmcShortagesGrouped->get($solId, collect());

            // Build unique target and source ID lists (dc_id, ec_id, h_id, tmc_id)
            $dc_id_list = $pFlows->pluck('source_id')->unique()->values()->toArray();
            $ec_id_list = $pFlows->pluck('target_id')->unique()->values()->toArray();
            $h_id_list = $hAllocations->pluck('target_id')->unique()->values()->toArray();
            $tmc_id_list = $tAllocations->pluck('target_id')->unique()->values()->toArray();

            // Process package flows with geometry
            $packageFlowRecords = $pFlows->map(function ($pf) use ($getGeometry) {
                return [
                    "source_id" => (int) $pf->source_id,
                    "target_id" => (int) $pf->target_id,
                    "geometry" => $getGeometry($pf->source_id, $pf->target_id, "ground"),
                    "flow" => (float) $pf->flow,
                    "flow_cost" => (float) $pf->flow_cost,
                ];
            })->values()->toArray();

            // Process shelter allocations with geometry
            $shelterAllocationRecords = $sAllocations->map(function ($sa) use ($getGeometry) {
                return [
                    "source_id" => (int) $sa->source_id,
                    "target_id" => (int) $sa->target_id,
                    "geometry" => $getGeometry($sa->source_id, $sa->target_id, "air"),
                    "flow" => (float) $sa->flow,
                ];
            })->values()->toArray();

            // Process hospital allocations with geometry
            $hospitalAllocationRecords = $hAllocations->map(function ($ha) use ($getGeometry) {
                $isGround = ($ha->g_flow_severe > 0 || $ha->g_flow_moderate > 0);
                return [
                    "source_id" => (int) $ha->source_id,
                    "target_id" => (int) $ha->target_id,
                    "geometry" => $getGeometry($ha->source_id, $ha->target_id, $isGround ? "ground" : "air"),
                    "g_flow_severe" => (float) $ha->g_flow_severe,
                    "num_gv_severe" => (float) $ha->num_gv_severe,
                    "g_flow_cost_severe" => (float) $ha->g_flow_cost_severe,
                    "a_flow_severe" => (float) $ha->a_flow_severe,
                    "num_av_severe" => (float) $ha->num_av_severe,
                    "a_flow_cost_severe" => (float) $ha->a_flow_cost_severe,
                    "g_flow_moderate" => (float) $ha->g_flow_moderate,
                    "num_gv_moderate" => (float) $ha->num_gv_moderate,
                    "g_flow_cost_moderate" => (float) $ha->g_flow_cost_moderate,
                    "a_flow_moderate" => (float) $ha->a_flow_moderate,
                    "num_av_moderate" => (float) $ha->num_av_moderate,
                    "a_flow_cost_moderate" => (float) $ha->a_flow_cost_moderate,
                ];
            })->values()->toArray();

            // Process TMC allocations with geometry
            $tmcAllocationRecords = $tAllocations->map(function ($ta) use ($getGeometry) {
                $isGround = ($ta->g_flow_moderate > 0);
                return [
                    "source_id" => (int) $ta->source_id,
                    "target_id" => (int) $ta->target_id,
                    "geometry" => $getGeometry($ta->source_id, $ta->target_id, $isGround ? "ground" : "air"),
                    "g_flow_moderate" => (float) $ta->g_flow_moderate,
                    "num_gv_moderate" => (float) $ta->num_gv_moderate,
                    "g_flow_cost_moderate" => (float) $ta->g_flow_cost_moderate,
                    "a_flow_moderate" => (float) $ta->a_flow_moderate,
                    "num_av_moderate" => (float) $ta->num_av_moderate,
                    "a_flow_cost_moderate" => (float) $ta->a_flow_cost_moderate,
                ];
            })->values()->toArray();

            // Structure shortages
            $solution_shelter_shortage = $sShortages->pluck('shortage', 'node_id')->toArray();
            $solution_hospital_shortage_severe = $hShortages->pluck('severe_shortage', 'node_id')->toArray();
            $solution_hospital_shortage_moderate = $hShortages->pluck('moderate_shortage', 'node_id')->toArray();
            $solution_tmc_shortage = $tShortages->pluck('shortage', 'node_id')->toArray();

            // Format costs
            $costs = $costRow ? [
                "package_flow_cost" => (float) $costRow->package_flow_cost,
                "package_cost" => (float) $costRow->package_cost,
                "ground_vehicle_cost" => (float) $costRow->ground_vehicle_cost,
                "air_vehicle_cost" => (float) $costRow->air_vehicle_cost,
                "shelter_establish_cost" => (float) $costRow->shelter_establish_cost,
                "tmc_establish_cost" => (float) $costRow->tmc_establish_cost,
                "total_cost" => (float) $costRow->total_cost,
            ] : null;

            // Construct exact _decode_one schema dictionary
            $decodedSolutions[] = [
                "solution_id" => (int) $solId,
                "dc_id" => $dc_id_list,
                "ec_id" => $ec_id_list,
                "h_id" => $h_id_list,
                "tmc_id" => $tmc_id_list,
                "F1" => (float) $sol->z1,
                "F2" => (float) $sol->z2,
                "F3" => (float) $sol->z3,
                "package_flows" => $packageFlowRecords,
                "shelter_allocations" => $shelterAllocationRecords,
                "hospital_allocations" => $hospitalAllocationRecords,
                "tmc_allocations" => $tmcAllocationRecords,
                "solution_shelter_shortage" => $solution_shelter_shortage,
                "solution_hospital_shortage_severe" => $solution_hospital_shortage_severe,
                "solution_hospital_shortage_moderate" => $solution_hospital_shortage_moderate,
                "solution_tmc_shortage" => $solution_tmc_shortage,
                "costs" => $costs
            ];
        }

        return [
            'success' => true,
            'scenario' => $scenario,
            'data' => $decodedSolutions
        ];
    }
}
