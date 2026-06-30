<?php

namespace App\Services;
 
use App\Models\Scenario;
 
class ScenarioService
{
    public function getActiveScenarioId(): int
    {
        return Scenario::where('active', 1)->value('id');
    }
}