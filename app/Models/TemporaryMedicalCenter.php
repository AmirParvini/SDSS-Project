<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class TemporaryMedicalCenter extends Model
{
    protected $fillable = ['capacity'];

    public function node()
    {
        return $this->belongsTo(Node::class, 'node_id', 'id');
    }
}
