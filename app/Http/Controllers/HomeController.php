<?php

namespace App\Http\Controllers;
use App\Models\HscParameter;
use App\Services\ScenarioService;
use App\Services\NodeDataService;
use App\Support\GeoJsonConverter;

class HomeController extends Controller
{

    public function __construct(
        private ScenarioService $scenarioService,
        private NodeDataService $nodeDataService,
        private GeoJsonConverter $geoJsonConverter
    ) {}

    public function index()
    {
        $scenarioId = $this->scenarioService->getActiveScenarioId();

        $nodesByType = $this->nodeDataService->getNodesForScenario($scenarioId);

        $hscParameters = HscParameter::where('scenario_id', $scenarioId)->get();

        $geojsonNodes = $this->geoJsonConverter->convert($nodesByType);

        return response()->json([
            'hsc_parameters' => $hscParameters,
            'geojson_nodes' => $geojsonNodes,
        ]);
    }
}
