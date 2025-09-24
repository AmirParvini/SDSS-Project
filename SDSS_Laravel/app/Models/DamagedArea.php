<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class DamagedArea extends Model
{
    protected $fillable = ['name', 'injured', 'lat', 'lng'];
}
