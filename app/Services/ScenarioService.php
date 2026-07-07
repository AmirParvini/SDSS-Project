<?php

namespace App\Services;
 
use App\Models\Scenario;
 
class ScenarioService
{
    static function selectScenario(int $current_scenario_id){

        // First, set all scenarios to inactive
        Scenario::query()->update(['active' => 0]);
        
        // Then, set the current scenario to active
        Scenario::where('id', $current_scenario_id)->update(['active' => 1]);

    }

    public function getActiveScenarioId(): int
    {
        return Scenario::where('active', 1)->value('id');
    }
}