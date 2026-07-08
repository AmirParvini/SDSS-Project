<?php

namespace App\Http\Controllers;

use App\Models\HscParameter;
use App\Models\Scenario;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Log;

class HscParametersController extends Controller
{
    /**
     * Display a listing of the resource.
     */
    public function index() {}

    /**
     * Show the form for creating a new resource.
     */
    public function create()
    {
        //
    }

    /**
     * Store a newly created resource in storage.
     */
    public function store(Request $request)
    {
        Log::info($request->all());
        $hsc_params_model = new HscParameter();
        $hsc_params_model->fill($request->all());
        $hsc_params_model->scenario_id = Scenario::where('active', 1)->value('id');
        $hsc_params_model->save();
        return response()->json(["success" => true]);
    }

    /**
     * Display the specified resource.
     */
    public function show(int $scenario_id) {}

    /**
     * Show the form for editing the specified resource.
     */
    public function edit(int $scenario_id)
    {
        $hsc_params = HscParameter::where('scenario_id', $scenario_id)->get();
        return response()->json(["hsc_parameters" => $hsc_params[0]]);
    }

    /**
     * Update the specified resource in storage.
     */
    public function update(Request $request, int $scenario_id)
    {
        Log::info($request->all());
        HscParameter::query()->where('scenario_id', $scenario_id)->update($request->all());
        return response()->json(["success" => true]);
    }

    /**
     * Remove the specified resource from storage.
     */
    public function destroy(string $id)
    {
        //
    }
}
