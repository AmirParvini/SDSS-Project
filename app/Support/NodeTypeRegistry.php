<?php

namespace App\Support;

use App\Models\Shelter;

class NodeTypeRegistry
{
    public static function definitions(): array
    {
        return [
            'ec' => [
                'model' => \App\Models\Shelter::class,
                'table' => 'shelters',
                'select' => ['shelters.area'],
            ],
            'dc' => [
                'model' => \App\Models\DistributionCenter::class,
                'table' => 'distribution_centers',
                'select' => [],
            ],
            'da' => [
                'model' => \App\Models\AffectedArea::class,
                'table' => 'affected_areas',
                'select' => ['affected_areas.affected_pop'],
            ],
            'tmc' => [
                'model' => \App\Models\TemporaryMedicalCenter::class,
                'table' => 'temporary_medical_centers',
                'select' => ['temporary_medical_centers.capacity'],
            ],
            'h' => [
                'model' => \App\Models\Hospital::class,
                'table' => 'hospitals',
                'select' => ['hospitals.capacity'],
            ],
        ];
    }
}