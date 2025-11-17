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
            $data = ['pathes' => [
                    'idc_ec_path' => IDCtoEcPath::all(),
                    'da_h_path'   => DAtoHospitalPath::all(),
                    'da_ec_dist'   => DAtoEcDist::all(),
                    'da_tmc_path' => DAtoTmcPath::all()
                ]];

            // Save data to JSON file
            $filePath = 'C:/Users/Amir/Desktop/SDSS-Project/Python_Codes/Python_Codes\NSGA_II_HSC/exports/HSC_Parameters.json';
            $directory = dirname($filePath);
            if (!is_dir($directory)) {
                mkdir($directory, 0755, true);
            }
            file_put_contents($filePath, json_encode($data, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE));

            return response()->json($data);
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
