<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class HscParameter extends Model
{
    protected $fillable = [
        "budget",
        "t1",
        "t2",
        "pua",
        "rta",
        "rtc",
        "gv_speed",
        "av_speed",
        "phi_min_s",
        "phi_max_s",
        "ks_s",
        "tm_s",
        "phi_min_m",
        "phi_max_m",
        "ks_m",
        "tm_m",
        "itst",
        "wt",
        "rp_cost",
        "rpt_cost",
        "tmc_cost",
        "shelter_cost",
        "gv_cost",
        "av_cost",
        "gv_severe_capacity",
        "av_severe_capacity",
        "gv_moderate_capacity",
        "av_moderate_capacity",
    ];
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
