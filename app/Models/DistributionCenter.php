<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class DistributionCenter extends Model
{
    public function node()
    {
        return $this->belongsTo(Node::class, 'node_id', 'id');
    }
}
