<?php

namespace App\Services\Optimization;

use App\Models\Cost;
use App\Models\HospitalAllocation;
use App\Models\HospitalShortage;
use App\Models\PackageFlow;
use App\Models\ParetoSolution;
use App\Models\Path;
use App\Models\ShelterAllocation;
use App\Models\TemporaryMedicalCenterAllocation;
use App\Models\TmcShortage;
use App\Models\UnsettledPopulation;
use Throwable;

class SaveResults
{

    function store(array $results, int $scenario_id)
    {
        try{
            foreach ($results as $solution) {
                ParetoSolution::insert([
                    'solution_id' => $solution['solution_id'],
                    'scenario_id' => $scenario_id,
                    'z1' => $solution['F1'],
                    'z2' => $solution['F2'],
                    'z3' => $solution['F3'],
                ]);
                foreach ($solution['package_flows'] as $pf) {
                    PackageFlow::insert([
                        'scenario_id' => $scenario_id,
                        'solution_id' => $solution['solution_id'],
                        'source_id' => $pf['source_id'],
                        'target_id' => $pf['target_id'],
                        'flow' => $pf['flow'],
                        'flow_cost' => $pf['flow_cost'],
                        'unmet_demand' => $pf['unmet_demand'],
                    ]);
                }
                foreach ($solution['tmc_allocations'] as $ta) {
                    TemporaryMedicalCenterAllocation::insert([
                        'scenario_id' => $scenario_id,
                        'solution_id' => $solution['solution_id'],
                        'source_id' => $ta['source_id'],
                        'target_id' => $ta['target_id'],
                        'g_flow_moderate' => $ta['g_flow_moderate'],
                        'num_gv_moderate' => $ta['num_gv_moderate'],
                        'g_flow_cost_moderate' => $ta['g_flow_cost_moderate'],
                        'a_flow_moderate' => $ta['a_flow_moderate'],
                        'num_av_moderate' => $ta['num_av_moderate'],
                        'a_flow_cost_moderate' => $ta['a_flow_cost_moderate'],
                    ]);
                }
                foreach ($solution['hospital_allocations'] as $ha) {
                    HospitalAllocation::insert([
                        'scenario_id' => $scenario_id,
                        'solution_id' => $solution['solution_id'],
                        'source_id' => $ha['source_id'],
                        'target_id' => $ha['target_id'],
                        'g_flow_severe' => $ha['g_flow_severe'] ?? 0,
                        'num_gv_severe' => $ha['num_gv_severe'] ?? 0,
                        'g_flow_cost_severe' => $ha['g_flow_cost_severe'] ?? 0,
                        'a_flow_severe' => $ha['a_flow_severe'] ?? 0,
                        'num_av_severe' => $ha['num_av_severe'] ?? 0,
                        'a_flow_cost_severe' => $ha['a_flow_cost_severe'] ?? 0,
                        'g_flow_moderate' => $ha['g_flow_moderate'] ?? 0,
                        'num_gv_moderate' => $ha['num_gv_moderate']  ?? 0,
                        'g_flow_cost_moderate' => $ha['g_flow_cost_moderate']  ?? 0,
                        'a_flow_moderate' => $ha['a_flow_moderate']  ?? 0,
                        'num_av_moderate' => $ha['num_av_moderate']  ?? 0,
                        'a_flow_cost_moderate' => $ha['a_flow_cost_moderate']  ?? 0,
                    ]);
                }
                foreach ($solution['shelter_allocations'] as $sa) {
                    $source_id = $sa['source_id'];
                    $target_id = $sa['target_id'];
                    ShelterAllocation::insert([
                        'scenario_id' => $scenario_id,
                        'solution_id' => $solution['solution_id'],
                        'source_id' => $source_id,
                        'target_id' => $target_id,
                        'flow' => $sa['flow'],
                        'distance' => Path::query()->where("source_id", $source_id)->where("target_id", $target_id)
                            ->where("path_type", "air")->value("distance")
                    ]);
                }
                Cost::insert([
                    'scenario_id' => $scenario_id,
                    'solution_id' => $solution['solution_id'],
                    'package_flow_cost' => $solution['costs']['package_flow_cost'],
                    'package_cost' => $solution['costs']['package_cost'],
                    'ground_vehicle_cost' => $solution['costs']['ground_vehicle_cost'],
                    'air_vehicle_cost' => $solution['costs']['air_vehicle_cost'],
                    'shelter_establish_cost' => $solution['costs']['shelter_establish_cost'],
                    'tmc_establish_cost' => $solution['costs']['tmc_establish_cost'],
                    'total_cost' => $solution['costs']['total_cost']
                ]);
    
                $up = $solution['solution_unsettled_population'];
                foreach ($up as $id=>$pop)
                UnsettledPopulation::insert([
                    'scenario_id' => $scenario_id,
                    'solution_id' => $solution['solution_id'],
                    'node_id' => $id,
                    'pop' => $pop
                ]);
    
                $ts = $solution['solution_tmc_shortage'];
                foreach ($ts as $id=>$shortage)
                TmcShortage::insert([
                    'scenario_id' => $scenario_id,
                    'solution_id' => $solution['solution_id'],
                    'node_id' => $id,
                    'shortage' => $shortage
                ]);
    
                $hs = [];
                $hss = $solution['solution_hospital_shortage_severe'];
                $hsm = $solution['solution_hospital_shortage_moderate'];
                foreach ($hss as $id => $shortage){
                    $hs["$id"]['severe'] = $shortage;
                }
                foreach ($hsm as $id => $shortage){
                    $hs["$id"]['moderate'] = $shortage;
                }
                foreach ($hs as $id => $shortage){
                    HospitalShortage::insert([
                    'scenario_id' => $scenario_id,
                    'solution_id' => $solution['solution_id'],
                    'node_id' => (int)$id,
                    'severe_shortage' => $shortage['severe'] ?? 0,
                    'moderate_shortage' => $shortage['moderate'] ?? 0
                ]);
                }
                
            }
        }catch (Throwable $e) {
            throw $e;
        }
    }
}
