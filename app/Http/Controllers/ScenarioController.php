<?php

namespace App\Http\Controllers;

use App\Models\Scenario;
use Illuminate\Http\Request;
use App\Services\ScenarioService;
use App\Services\NodeDataService;
use App\Support\GeoJsonConverter;
use Illuminate\Support\Facades\Log;
use Symfony\Component\Console\Descriptor\ReStructuredTextDescriptor;

class ScenarioController extends Controller
{
    function select_scenario(Request $request)
    {
        ScenarioService::selectScenario($request->current_scenario_id);
        return response()->json(["success" => true]);
    }

    function index() {
        $scenarios = Scenario::all();
        $activeId = Scenario::where('active', 1)->value('id');
        return response()->json(["scenarios" => $scenarios, "activeId" => $activeId]);
    }

    function store(Request $request) {
        $scenario = new Scenario();
        $scenario->name = $request->name;
        $scenario->description = $request->description;
        $scenario->active = 0;
        $scenario->save();
        ScenarioService::selectScenario($scenario->id);
        return response()->json(["success" => true]);
    }

    function update(Request $request, int $id) {
        Scenario::query()->where('id', $id)->update($request->all());
        return response()->json(["success" => true]);
    }
}
