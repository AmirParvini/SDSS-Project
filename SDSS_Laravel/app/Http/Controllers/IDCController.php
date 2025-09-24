<?php

namespace App\Http\Controllers;

use App\Models\IDC;
use Illuminate\Http\Request;

class IDCController extends Controller
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
            IDC::create([
                'name' => $request->name,
                'capacity' => $request->capacity,
                'fixed_cost' => $request->cost,
                'lat' => $request->lat,
                'lng' => $request->lng
            ]);

            return response()->json(['success' => true]);
        } catch (\Illuminate\Validation\ValidationException $e) {
            return response()->json(['success' => false, 'errors' => $e->errors()], 422);
        } catch (\Exception $e) {
            return response()->json(['success' => false, 'message' => $e->getMessage()], 500);
        }
    }

    /**
     * Display the specified resource.
     */
    public function show(IDC $iDC)
    {
        //
    }

    /**
     * Show the form for editing the specified resource.
     */
    public function edit(IDC $iDC)
    {
        //
    }

    /**
     * Update the specified resource in storage.
     */
    public function update(Request $request, IDC $iDC)
    {
        //
    }

    /**
     * Remove the specified resource from storage.
     */
    public function destroy(IDC $iDC)
    {
        //
    }
}
