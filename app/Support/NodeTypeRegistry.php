<?php

namespace App\Support;
 
class NodeTypeRegistry
{
    public static function definitions(): array
    {
        return [
            'shelter' => [
                'table' => 'shelters',
                'select' => ['shelters.area'],
            ],
            'distribution_center' => [
                'table' => 'distribution_centers',
                'select' => [],
            ],
            'affected_area' => [
                'table' => 'affected_areas',
                'select' => ['affected_areas.affected_pop'],
            ],
            'temporary_medical_center' => [
                'table' => 'temporary_medical_centers',
                'select' => ['temporary_medical_centers.capacity'],
            ],
            'hospital' => [
                'table' => 'hospitals',
                'select' => ['hospitals.capacity'],
            ],
        ];
    }
}