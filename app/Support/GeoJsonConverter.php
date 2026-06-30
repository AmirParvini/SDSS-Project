<?php

namespace App\Support;
 
class GeoJsonConverter
{
    public function convert(array $nodesByType): array
    {
        $features = [];
 
        foreach ($nodesByType as $items) {
            foreach ($items as $properties) {
                $geometry = $properties['geometry'];
                unset(
                    $properties['scenario_id'],
                    $properties['geometry'],
                    $properties['created_at'],
                    $properties['updated_at']
                );
 
                $features[] = [
                    'type' => 'Feature',
                    'geometry' => $geometry,
                    'properties' => $properties,
                ];
            }
        }
 
        return [
            'type' => 'FeatureCollection',
            'features' => $features,
        ];
    }
}
