<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Node extends Model
{
    protected $fillable = ['type', 'name', 'geometry'];

    protected $casts = [
        'geometry' => 'array',
        'area' => 'float',
        'capacity' => 'int',
        'affected_pop' => 'int'
    ];
}
