<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use Illuminate\Support\Facades\DB;

/**
 * Imports air/ground distance & geometry data from HSC_Parameters.json
 * into the `paths` table.
 *
 * path_type values written to the DB ('ground' or 'air'):
 *   - idc_ec_path : 1 row,  'ground' (real road geometry exists)
 *   - da_h_path   : 2 rows, 'ground' (road geometry) + 'air' (distance_helicopter, straight line)
 *   - da_ec_dist  : 1 row,  'air'    (no road path in source data, only straight distance)
 *   - da_tmc_path : 2 rows, 'ground' (road geometry) + 'air' (distance_helicopter, straight line)
 *
 * The JSON file uses the OLD node IDs (idc_id, da_id, h_id, tmc_id, ec_id).
 * These are remapped to the NEW global node IDs used in the current
 * database (see the *_OLD_IDS / *_NEW_IDS arrays below) BEFORE insertion.
 *
 * Usage:
 *   php artisan paths:import
 *   php artisan paths:import /full/path/to/HSC_Parameters.json
 */
class ImportHscPaths extends Command
{
    protected $signature = 'paths:import {file=HSC_Parameters.json}';

    protected $description = 'Import distances/geometries from HSC_Parameters.json into the paths table (with ID remapping)';

    /**
     * --- ID REMAPPING TABLES -------------------------------------------------
     * Left  = old IDs as they appear inside HSC_Parameters.json
     * Right = new IDs that must be written into paths.source_id / target_id
     *
     * IMPORTANT ASSUMPTION: old IDs are mapped to new IDs by matching
     * ascending order (1st old id -> 1st new id, 2nd -> 2nd, ...).
     * Adjust the *_OLD_IDS arrays below if the real correspondence differs.
     */
    private const DC_OLD_IDS  = [1, 2, 3];               // idc_id in JSON
    private const DC_NEW_IDS  = [1, 2, 3];

    private const DA_OLD_IDS  = [3, 4, 5, 6, 7];          // da_id in JSON
    private const DA_NEW_IDS  = [29, 30, 31, 32, 33];

    private const H_OLD_IDS   = [1, 2, 3, 4];             // h_id in JSON
    private const H_NEW_IDS   = [4, 5, 6, 7];

    private const TMC_OLD_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]; // tmc_id in JSON
    private const TMC_NEW_IDS = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28];

    private const EC_OLD_IDS  = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]; // ec_id in JSON
    private const EC_NEW_IDS  = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18];

    private array $dcMap;
    private array $daMap;
    private array $hMap;
    private array $tmcMap;
    private array $ecMap;

    public function __construct()
    {
        parent::__construct();

        $this->dcMap  = array_combine(self::DC_OLD_IDS, self::DC_NEW_IDS);
        $this->daMap  = array_combine(self::DA_OLD_IDS, self::DA_NEW_IDS);
        $this->hMap   = array_combine(self::H_OLD_IDS, self::H_NEW_IDS);
        $this->tmcMap = array_combine(self::TMC_OLD_IDS, self::TMC_NEW_IDS);
        $this->ecMap  = array_combine(self::EC_OLD_IDS, self::EC_NEW_IDS);
    }

    public function handle(): int
    {
        $filePath = $this->resolveFilePath($this->argument('file'));

        if (!$filePath || !file_exists($filePath)) {
            $this->error("File not found: {$this->argument('file')}");
            return self::FAILURE;
        }

        $json = json_decode(file_get_contents($filePath), true, flags: JSON_THROW_ON_ERROR);
        $pathes = $json['pathes'] ?? null;

        if (!$pathes) {
            $this->error('Invalid JSON structure: "pathes" key was not found.');
            return self::FAILURE;
        }

        // The `geometry` column is NOT NULL, so every row needs a geometry value.
        // For da_ec_dist rows there is no path array in the JSON (it's an air
        // distance only), so we derive the DA and EC point coordinates from
        // the OTHER path arrays (where they appear as fixed endpoints) and
        // build a straight 2-point LineString between them.
        $daCoords = $this->extractEndpointCoords($pathes['da_h_path'] ?? [], 'da_id', first: true);
        $ecCoords = $this->extractEndpointCoords($pathes['idc_ec_path'] ?? [], 'ec_id', first: false);
        $hCoords = $this->extractEndpointCoords($pathes['da_h_path'] ?? [], 'h_id', first: false);
        $tmcCoords = $this->extractEndpointCoords($pathes['da_tmc_path'] ?? [], 'tmc_id', first: false);

        $rows = [];

        // 1) IDC (DC) <-> EC : ground path with geometry
        foreach ($pathes['idc_ec_path'] ?? [] as $item) {
            $rows[] = $this->buildRow(
                $this->mapId($this->dcMap, $item['idc_id'], 'dc'),
                $this->mapId($this->ecMap, $item['ec_id'], 'ec'),
                'ground',
                $item['distance'],
                $this->toGeoJson($this->parseCoordinates($item['path'])),
            );
        }

        // 2) DA <-> H : ground path with geometry, plus an 'air' (helicopter) row
        foreach ($pathes['da_h_path'] ?? [] as $item) {
            $sourceId = $this->mapId($this->daMap, $item['da_id'], 'da');
            $targetId = $this->mapId($this->hMap, $item['h_id'], 'h');

            $rows[] = $this->buildRow(
                $sourceId,
                $targetId,
                'ground',
                $item['distance'],
                $this->toGeoJson($this->parseCoordinates($item['path'])),
            );

            if (isset($item['distance_helicopter']) && $item['distance_helicopter'] !== null) {
                $rows[] = $this->buildRow(
                    $sourceId,
                    $targetId,
                    'air',
                    $item['distance_helicopter'],
                    $this->toGeoJson($this->buildStraightLine(
                        $daCoords[$item['da_id']] ?? null,
                        $hCoords[$item['h_id']] ?? null,
                        "da_id={$item['da_id']} / h_id={$item['h_id']}",
                    )),
                );
            }
        }

        // 3) DA <-> EC : air distance only (no road path exists) -> straight line
        foreach ($pathes['da_ec_dist'] ?? [] as $item) {
            $rows[] = $this->buildRow(
                $this->mapId($this->daMap, $item['da_id'], 'da'),
                $this->mapId($this->ecMap, $item['ec_id'], 'ec'),
                'air',
                $item['distance'],
                $this->toGeoJson($this->buildStraightLine(
                    $daCoords[$item['da_id']] ?? null,
                    $ecCoords[$item['ec_id']] ?? null,
                    "da_id={$item['da_id']} / ec_id={$item['ec_id']}",
                )),
            );
        }

        // 4) DA <-> TMC : ground path with geometry, plus an 'air' (helicopter) row
        foreach ($pathes['da_tmc_path'] ?? [] as $item) {
            $sourceId = $this->mapId($this->daMap, $item['da_id'], 'da');
            $targetId = $this->mapId($this->tmcMap, $item['tmc_id'], 'tmc');

            $rows[] = $this->buildRow(
                $sourceId,
                $targetId,
                'ground',
                $item['distance'],
                $this->toGeoJson($this->parseCoordinates($item['path'])),
            );

            if (isset($item['distance_helicopter']) && $item['distance_helicopter'] !== null) {
                $rows[] = $this->buildRow(
                    $sourceId,
                    $targetId,
                    'air',
                    $item['distance_helicopter'],
                    $this->toGeoJson($this->buildStraightLine(
                        $daCoords[$item['da_id']] ?? null,
                        $tmcCoords[$item['tmc_id']] ?? null,
                        "da_id={$item['da_id']} / tmc_id={$item['tmc_id']}",
                    )),
                );
            }
        }

        $inserted = 0;
        foreach (array_chunk($rows, 500) as $chunk) {
            DB::table('paths')->insert($chunk);
            $inserted += count($chunk);
        }

        $this->info("Inserted {$inserted} rows into the `paths` table.");

        return self::SUCCESS;
    }

    /**
     * Looks for the file relative to the project base path, then as an
     * absolute path, falling back to storage/app.
     */
    private function resolveFilePath(string $file): ?string
    {
        $candidates = [
            $file,
            base_path($file),
            storage_path('app/' . $file),
        ];

        foreach ($candidates as $candidate) {
            if (file_exists($candidate)) {
                return $candidate;
            }
        }

        return null;
    }

    private function mapId(array $map, int $oldId, string $label): int
    {
        if (!array_key_exists($oldId, $map)) {
            throw new \RuntimeException("No mapping found for {$label} id {$oldId}");
        }

        return $map[$oldId];
    }

    private function buildRow(int $sourceId, int $targetId, string $pathType, string|float $distance, string $geoJson): array
    {
        return [
            'source_id'  => $sourceId,
            'target_id'  => $targetId,
            'path_type'  => $pathType,
            'distance'   => round((float) $distance, 2),
            'geometry'   => $geoJson,
            'created_at' => now(),
            'updated_at' => now(),
        ];
    }

    /**
     * Parses a Postgres `path` literal, e.g.
     *   {"(51.41,35.70)","(51.42,35.71)", ...}
     * into an array of [lon, lat] pairs. Source data is already (lon, lat),
     * which matches GeoJSON coordinate order, so no swap is needed.
     */
    private function parseCoordinates(string $rawPath): array
    {
        preg_match_all('/\(([-\d.]+),([-\d.]+)\)/', $rawPath, $matches);

        $coordinates = [];
        foreach ($matches[1] as $i => $lon) {
            $coordinates[] = [(float) $lon, (float) $matches[2][$i]];
        }

        return $coordinates;
    }

    /**
     * Builds a [old_id => [lon, lat]] lookup table from a list of path
     * records, taking either the first or last coordinate of each path
     * as the fixed endpoint for the given id field (e.g. 'da_id', 'ec_id').
     * Only the first occurrence per id is used (the endpoint is constant
     * across all paths sharing that id, verified against the source data).
     */
    private function extractEndpointCoords(array $items, string $idField, bool $first): array
    {
        $coords = [];

        foreach ($items as $item) {
            $id = $item[$idField];

            if (isset($coords[$id])) {
                continue;
            }

            $points = $this->parseCoordinates($item['path']);

            if (empty($points)) {
                continue;
            }

            $coords[$id] = $first ? $points[0] : $points[array_key_last($points)];
        }

        return $coords;
    }

    /**
     * Builds a 2-point [from, to] coordinate array for a straight ('air')
     * line. Returns an empty array (and logs a warning) if either endpoint
     * could not be resolved, since the `geometry` column is NOT NULL.
     */
    private function buildStraightLine(?array $from, ?array $to, string $context): array
    {
        if ($from === null || $to === null) {
            $this->warn("Could not resolve coordinates for {$context}, storing an empty geometry.");
            return [];
        }

        return [$from, $to];
    }

    /**
     * Wraps an array of [lon, lat] coordinate pairs into a GeoJSON
     * LineString string. The `geometry` column is NOT NULL, so when no
     * coordinates could be resolved this returns an empty LineString
     * rather than null.
     */
    private function toGeoJson(array $coordinates): string
    {
        return json_encode([
            'type'        => 'LineString',
            'coordinates' => $coordinates,
        ], JSON_UNESCAPED_UNICODE);
    }
}
