<?php

namespace App\Http\Controllers;

use App\Models\Scenario;
use App\Support\NodeTypeRegistry;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Log;

class NodeCrudController extends Controller
{
    public function store(Request $request)
    {
        $validated = $request->validate([
            'type' => 'required|string',
            'name' => 'required|string',
            'lat' => 'required|numeric',
            'lng' => 'required|numeric',
            'scenario_id' => 'nullable|integer',
        ]);

        $type = $request->type;
        $registry = NodeTypeRegistry::definitions();

        if (!isset($registry[$type])) {
            return response()->json(['success' => false, 'message' => 'Invalid node type: ' . $request->type], 400);
        }

        $table = $registry[$type]['table'];

        DB::beginTransaction();
        $activeScenario = Scenario::where('active', 1)->first();
        try {
            $nodeId = DB::table('nodes')->insertGetId([
                'scenario_id' => $activeScenario->id,
                'type' => $type,
                'name' => $request->name,
                'geometry' => json_encode([
                    'type' => 'Point',
                    'coordinates' => [(float)$request->lng, (float)$request->lat]
                ]),
                'created_at' => now(),
                'updated_at' => now(),
            ]);

            $model_class = $registry[$type]['model'];
            $fields = $model_class ? (new $model_class)->getFillable() : [];
            $specificData = $request->only($fields);
            $specificData['node_id'] = $nodeId;

            DB::table($table)->insert($specificData);

            DB::commit();
            return response()->json(['success' => true, 'node_id' => $nodeId]);
        } catch (\Exception $e) {
            DB::rollBack();
            return response()->json(['success' => false, 'message' => $e->getMessage()], 500);
        }
    }

    public function update(Request $request, $id)
    {
        $validated = $request->validate([
            'type' => 'required|string',
            'name' => 'required|string',
            'lat' => 'required|numeric',
            'lng' => 'required|numeric',
        ]);

        $type = $request->type;
        $registry = NodeTypeRegistry::definitions();

        if (!isset($registry[$type])) {
            return response()->json(['success' => false, 'message' => 'Invalid node type'], 400);
        }

        $table = $registry[$type]['table'];

        DB::beginTransaction();
        try {
            DB::table('nodes')->where('id', $id)->update([
                'name' => $request->name,
                'geometry' => json_encode([
                    'type' => 'Point',
                    'coordinates' => [(float)$request->lng, (float)$request->lat]
                ]),
                'updated_at' => now(),
            ]);

            $model_class = $registry[$type]['model'];
            $fields = $model_class ? (new $model_class)->getFillable() : [];
            $specificData = $request->only($fields);

            if (!empty($specificFields)) {
                $specificData = $request->only($specificFields);
                DB::table($table)->where('node_id', $id)->update($specificData);
            }

            DB::commit();
            return response()->json(['success' => true]);
        } catch (\Exception $e) {
            DB::rollBack();
            return response()->json(['success' => false, 'message' => $e->getMessage()], 500);
        }
    }

    public function destroy($id)
    {
        $node = DB::table('nodes')->where('id', $id)->first();
        if (!$node) {
            return response()->json(['success' => false, 'message' => 'Node not found'], 404);
        }

        $registry = NodeTypeRegistry::definitions();
        $type = $node->type;
        
        DB::beginTransaction();
        try {
            if (isset($registry[$type])) {
                $table = $registry[$type]['table'];
                DB::table($table)->where('node_id', $id)->delete();
            }

            DB::table('nodes')->where('id', $id)->delete();

            DB::commit();
            return response()->json(['success' => true]);
        } catch (\Exception $e) {
            DB::rollBack();
            return response()->json(['success' => false, 'message' => $e->getMessage()], 500);
        }
    }
}
