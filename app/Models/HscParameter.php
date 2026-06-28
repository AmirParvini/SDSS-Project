<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class HscParameter extends Model
{
    protected $casts = [
        "budget" => "float",
        "t1" => 'float',
        "t2" => "float",
        "pua" => "float",
        "rta" => "float",
        "phi_min_s" => "float",
        "phi_max_s" => "float",
        "ks_s" => "float",
        "phi_min_m" => "float",
        "phi_max_m" => "float",
        "ks_m" => "float",
        "rp_cost" => "float",
        "rpt_cost" => "float",
        "tmc_cost" => "float",
        "shelter_cost" => "float",
        "gv_cost" => "float",
        "av_cost" => "float"
    ];
}
