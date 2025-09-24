<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class IDC extends Model
{
    protected $fillable = ['name', 'capacity','fixed_cost', 'lat', 'lng'];
}
