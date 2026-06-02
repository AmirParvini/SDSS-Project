<?php

namespace App\Http\Controllers;

use App\Models\TMC;
use Illuminate\Http\Request;

class TMCController extends Controller
{
    /**
     * Display a listing of the resource.
     */
    public function index()
    {
        //
    }

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
        try {
            $tmc_point = TMC::create([
                'name' => $request->name,
                'capacity' => $request->capacity,
                'fixed_cost' => $request->cost,
                'lat' => $request->lat,
                'lng' => $request->lng
            ]);

            return response()->json(['success' => true, 'pointtype' => 'TMC', 'point' => $tmc_point]);
        } catch (\Illuminate\Validation\ValidationException $e) {
            return response()->json(['success' => false, 'errors' => $e->errors()], 422);
        } catch (\Exception $e) {
            return response()->json(['success' => false, 'message' => $e->getMessage()], 500);
        }
    }

    /**
     * Display the specified resource.
     */
    public function show(TMC $tMC)
    {
        //
    }

    /**
     * Show the form for editing the specified resource.
     */
    public function edit(TMC $tMC)
    {
        //
    }

    /**
     * Update the specified resource in storage.
     */
    public function update(Request $request, TMC $tMC)
    {
        //
    }

    /**
     * Remove the specified resource from storage.
     */
    public function destroy(Int $id)
    {
        try{
            $point = TMC::find($id);
            $point->delete();
            return response()->json(['success' => true]);
        }
        catch (\Exception $e) {
            return response()->json(['success' => false, 'message' => $e->getMessage()], 500);
        }
    }
}
