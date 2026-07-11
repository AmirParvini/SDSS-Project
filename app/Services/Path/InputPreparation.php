<?php

namespace App\Services\Path;

use App\Models\Node;
use App\Models\Scenario;
use App\Support\NodeTypeRegistry;

class InputPreparation
{
    public function __construct() {}

    function assinments(int $scenario_id)
    {
        $nodes = $this->_getData($scenario_id);
        $assing_key = [
            "dc" => ["ground" => ["ec"]],
            "da" => [
                "ground" => ["h", "tmc"],
                "air" => ["ec", "h", "tmc"]
            ]
        ];
        $ground_assignments = [];
        $air_assignments = [];
        foreach ($assing_key as $source_key => $type_groups) {
            if (empty($nodes[$source_key])) {
                continue;
            }
            foreach ($nodes[$source_key] as $source_node) {
                foreach ($type_groups as $type => $target_key_list) {
                    $targets = [];
                    foreach ($target_key_list as $target_key) {
                        if (!empty($nodes[$target_key])) {
                            $targets = array_merge($targets, $nodes[$target_key]);
                        }
                    }
                    $assignment = [
                        "source" => $source_node,
                        "targets" => $targets
                    ];
                    if ($type === "ground") {
                        $ground_assignments[] = $assignment;
                    } elseif ($type === "air") {
                        $air_assignments[] = $assignment;
                    }
                }
            }
        }
        return $this->filterAssignments($ground_assignments, $air_assignments);
    }

    public function filterAssignments(array $ground_assignments, array $air_assignments): array
    {
        $existing = \App\Models\Path::query()
            ->select('source_id', 'target_id', 'path_type')
            ->get();

        $groundMap = [];
        $airMap = [];

        foreach ($existing as $path) {
            if ($path->path_type === 'ground') {
                $groundMap[$path->source_id][$path->target_id] = true;
            } elseif ($path->path_type === 'air') {
                $airMap[$path->source_id][$path->target_id] = true;
            }
        }

        $filteredGround = $this->_filter($ground_assignments, $groundMap);
        $filteredAir = $this->_filter($air_assignments, $airMap);

        return [
            'ground' => $filteredGround,
            'air' => $filteredAir
        ];
    }

    private function _filter(array $assignments, array $lookupMap): array
    {
        $filtered = [];
        foreach ($assignments as $assignment) {
            $sourceId = $assignment['source']['id'];
            $newTargets = [];
            foreach ($assignment['targets'] as $target) {
                $targetId = $target['id'];
                if (!isset($lookupMap[$sourceId][$targetId])) {
                    $newTargets[] = $target;
                }
            }
            if (!empty($newTargets)) {
                $assignment['targets'] = $newTargets;
                $filtered[] = $assignment;
            }
        }
        return $filtered;
    }

    function _getData(int $scenario_id)
    {
        $nodes = [];
        foreach (NodeTypeRegistry::definitions() as $type => $props) {
            $type_nodes = Node::query()->where([
                "scenario_id" => $scenario_id,
                "type" => $type
            ])->get(["id", "geometry"])->toArray();
            $nodes[$type] = $type_nodes;
        }
        return $nodes;
    }
}
