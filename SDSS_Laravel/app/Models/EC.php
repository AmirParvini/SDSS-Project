<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class EC extends Model
{
    protected $fillable = ['name', 'demand', 'lat', 'lng'];
}
