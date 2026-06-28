<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class AffectedArea extends Model
{
    protected $fillable = ['population'];
    
    public function node()
    {
        return $this->belongsTo(Node::class, 'node_id', 'id');
    }
}
