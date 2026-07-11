<?php

namespace App\Services\Path;

use App\Services\Path\SaveResults;

use proj4php\Proj4php;
use proj4php\Proj;
use proj4php\Point;

class AirPathService
{

    public function __construct(
        private SaveResults $save_results
    ) {}

    function run(array $assigns)
    {
        $output = $this->_calculate($assigns);
        $this->save_results->store($output, "air");
        return $output;
    }

    function _calculate(array $assigns)
    {
        if (empty($assigns)) {
            return [];
        }

        // مقداردهی اولیه Proj4
        $proj4 = new Proj4php();
        $projWGS84 = new Proj('EPSG:4326', $proj4); // سیستم مختصات جغرافیایی
        $projUTM = new Proj('EPSG:32639', $proj4);  // نمونه: UTM Zone 39N (بسته به موقعیت تغییر کند)

        $output = [];
        foreach ($assigns as $assignment) {
            $source = $assignment['source'];
            $sourceId = $source['id'];
            $sourceCoords = $source['geometry']['coordinates'] ?? null;

            if (!$sourceCoords || count($sourceCoords) < 2) {
                continue;
            }

            foreach ($assignment['targets'] as $target) {
                $targetId = $target['id'];
                $targetCoords = $target['geometry']['coordinates'] ?? null;

                if (!$targetCoords || count($targetCoords) < 2) {
                    continue;
                }

                // ۱. تبدیل مختصات مبدا به UTM
                $pointSource = new Point($sourceCoords[0], $sourceCoords[1], $projWGS84);
                $utmSource = $proj4->transform($projUTM, $pointSource);
                $x1 = $utmSource->x;
                $y1 = $utmSource->y;

                // ۲. تبدیل مختصات مقصد به UTM
                $pointTarget = new Point($targetCoords[0], $targetCoords[1], $projWGS84);
                $utmTarget = $proj4->transform($projUTM, $pointTarget);
                $x2 = $utmTarget->x;
                $y2 = $utmTarget->y;

                // ۳. محاسبه فاصله بر حسب متر (فرمول فیثاغورس روی مختصات مسطح)
                $distance = sqrt(pow($x2 - $x1, 2) + pow($y2 - $y1, 2));

                // هندسه خروجی GeoJSON معمولاً باید همان WGS84 (طول و عرض جغرافیایی) باقی بماند
                $geometry = [
                    'type' => 'LineString',
                    'coordinates' => [
                        [(float)$sourceCoords[0], (float)$sourceCoords[1]],
                        [(float)$targetCoords[0], (float)$targetCoords[1]]
                    ]
                ];

                $output[] = [
                    'source_id' => $sourceId,
                    'target_id' => $targetId,
                    'distance' => $distance, // فاصله به متر
                    'geometry' => $geometry
                ];
            }
        }
        return $output;
    }
}
