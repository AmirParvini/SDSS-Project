<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class DamagedArea extends Model
{
    protected $fillable = ['name', 'affected_pop', 'lat', 'lng'];
}
