<?php

namespace App\Http\Controllers;

use App\Models\EC;
use Illuminate\Http\Request;

class ECController extends Controller
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
            $ec_point = EC::create([
                'name' => $request->name,
                'demand' => $request->demand,
                'fixed_cost' => $request->cost,
                'lat' => $request->lat,
                'lng' => $request->lng
            ]);

            return response()->json(['success' => true, 'pointtype' => 'EC', 'point' => $ec_point]);
        } catch (\Illuminate\Validation\ValidationException $e) {
            return response()->json(['success' => false, 'errors' => $e->errors()], 422);
        } catch (\Exception $e) {
            return response()->json(['success' => false, 'message' => $e->getMessage()], 500);
        }
    }

    /**
     * Display the specified resource.
     */
    public function show(EC $eC)
    {
        //
    }

    /**
     * Show the form for editing the specified resource.
     */
    public function edit(EC $eC)
    {
        //
    }

    /**
     * Update the specified resource in storage.
     */
    public function update(Request $request, EC $eC)
    {
        //
    }

    /**
     * Remove the specified resource from storage.
     */
    public function destroy(int $id)
    {
        try{
            $point = EC::find($id);
            $point->delete();
            return response()->json(['success' => true]);
        }
        catch (\Exception $e) {
            return response()->json(['success' => false, 'message' => $e->getMessage()], 500);
        }
    }
}
