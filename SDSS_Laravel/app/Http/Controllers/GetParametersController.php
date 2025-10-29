<?php

namespace App\Http\Controllers;

use App\Models\DamagedArea;
use App\Models\DAtoEcDist;
use App\Models\DAtoHospitalPath;
use App\Models\DAtoTmcPath;
use App\Models\EC;
use App\Models\Hospital;
use App\Models\IDC;
use App\Models\IDCtoEcPath;
use App\Models\TMC;
use Illuminate\Support\Facades\Cache;
use Illuminate\Http\Request;

class GetParametersController extends Controller
{
    /**
     * Display a listing of the resource.
     */
    public function index()
    {
        // if (Cache::has('pathes')) {
        //     $pathes = Cache::get('pathes');
        //     return response()->json(['pathes' => $pathes]);
        // } else {
        //     Cache::forever('pathes', [
        //         'idc_ec_path' => IDCtoEcPath::all(),
        //         'da_h_path'   => DAtoHospitalPath::all(),
        //         'da_ec_dist'   => DAtoEcDist::all(),
        //         'da_tmc_path' => DAtoTmcPath::all()
        //     ]);
        //     return response()->json(['pathes' => [
        //         'idc_ec_path' => IDCtoEcPath::all(),
        //         'da_h_path'   => DAtoHospitalPath::all(),
        //         'da_ec_dist'   => DAtoEcDist::all(),
        //         'da_tmc_path' => DAtoTmcPath::all()
        //     ]]);
        // }
        try {
            return response()->json(['pathes' => [
                    'idc_ec_path' => IDCtoEcPath::all(),
                    'da_h_path'   => DAtoHospitalPath::all(),
                    'da_ec_dist'   => DAtoEcDist::all(),
                    'da_tmc_path' => DAtoTmcPath::all()
                ]]);
        } catch (\Exception $e) {
            return response()->json(['error' => $e->getMessage()], 500);
        }
    }

    /**
     * Store a newly created resource in storage.
     */
    public function store(Request $request)
    {
        //
    }

    /**
     * Display the specified resource.
     */
    public function show(string $id)
    {
        //
    }

    /**
     * Update the specified resource in storage.
     */
    public function update(Request $request, string $id)
    {
        //
    }

    /**
     * Remove the specified resource from storage.
     */
    public function destroy(string $id)
    {
        //
    }
}
