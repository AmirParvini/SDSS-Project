<?php

namespace App\Http\Controllers;

use App\Models\DamagedArea;
use App\Models\EC;
use App\Models\Hospital;
use App\Models\IDC;
use App\Models\TMC;
use Illuminate\Http\Request;

class HomeController extends Controller
{
    function index() {
        $idc_points = IDC::all();
        $ec_points = EC::all();
        $da_points = DamagedArea::all();
        $tmc_points = TMC::all();
        $H_points = Hospital::all();
        return view('home', compact('idc_points', 'ec_points', 'da_points', 'tmc_points', 'H_points'));
    }
}
