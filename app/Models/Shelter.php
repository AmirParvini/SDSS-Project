<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Shelter extends Model
{
    protected $fillable = ['area'];

    public function node()
    {
        return $this->belongsTo(Node::class, 'node_id', 'id');
    }

    protected $casts = [
        'area' => 'float'
    ];
}
