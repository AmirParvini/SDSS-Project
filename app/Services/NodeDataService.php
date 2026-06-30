<?php

namespace App\Services;
 
use App\Models\Node;
use App\Support\NodeTypeRegistry;
 
class NodeDataService
{
    public function getNodesForScenario(int $scenarioId): array
    {
        $nodesByType = [];
 
        foreach (NodeTypeRegistry::definitions() as $type => $def) {
            $table = $def['table'];
            $select = array_merge(['nodes.*'], $def['select']);
 
            $nodesByType[$type] = Node::join($table, 'nodes.id', '=', "{$table}.node_id")
                ->where('nodes.scenario_id', $scenarioId)
                ->select($select)
                ->get()
                ->toArray();
        }
 
        return $nodesByType;
    }
}