<?php

namespace App\Http\Controllers;

use App\Models\DamagedArea;
use Illuminate\Http\Request;

class DamagedAreaController extends Controller
{
    public function store(Request $request)
    {
        try {
            // $request->validate([
            //     'name' => 'required|string|max:255',
            //     'population' => 'required|string|max:255',
            //     'lat' => 'required|string|max:255',
            //     'lng' => 'required|string|max:255',
            // ]);

            DamagedArea::create([
                'name' => $request->name,
                'injured' => $request->injured,
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
}
