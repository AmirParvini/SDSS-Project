<?php

namespace App\Services\optimization;

use App\Models\DistributionCenter;
use App\Models\Shelter;
use App\Models\AffectedArea;
use App\Models\Hospital;
use App\Models\TemporaryMedicalCenter;
use App\Models\HscParameter;
use App\Models\Path;

class BuildData
{
    protected array $dc_id;
    protected array $ec_id;
    protected array $da_id;
    protected array $h_id;
    protected array $tmc_id;

    protected array $hsc_parameters;
    protected array $affected_pops;
    protected array $shelters_area;
    protected array $hospitals_capacity;
    protected array $tmcs_capacity;
    protected array $distances;

    public function build(int $scenario_id): array
    {
        $this->dc_id = DistributionCenter::query()->whereRelation('node', 'scenario_id', $scenario_id)->pluck('node_id')->toArray();
        $this->ec_id = Shelter::query()->whereRelation('node', 'scenario_id', $scenario_id)->pluck('node_id')->toArray();
        $this->da_id = AffectedArea::query()->whereRelation('node', 'scenario_id', $scenario_id)->pluck('node_id')->toArray();
        $this->h_id = Hospital::query()->whereRelation('node', 'scenario_id', $scenario_id)->pluck('node_id')->toArray();
        $this->tmc_id = TemporaryMedicalCenter::query()->whereRelation('node', 'scenario_id', $scenario_id)->pluck('node_id')->toArray();
        $this->hsc_parameters = HscParameter::all()->where('scenario_id', $scenario_id)->toArray();
        $this->affected_pops = AffectedArea::query()->whereRelation('node', 'scenario_id', $scenario_id)->pluck('affected_pop', 'node_id')->toArray();
        $this->shelters_area = Shelter::query()->whereRelation('node', 'scenario_id', $scenario_id)->pluck('area', 'node_id')->toArray();
        $this->hospitals_capacity = Hospital::query()->whereRelation('node', 'scenario_id', $scenario_id)->pluck('capacity', 'node_id')->toArray();
        $this->tmcs_capacity = TemporaryMedicalCenter::query()->whereRelation('node', 'scenario_id', $scenario_id)->pluck('capacity', 'node_id')->toArray();
        // dd($this->shelters_area);
        return $this->build_data();
    }

    protected function build_data(): array
    {
        $dc_id = $this->dc_id;
        $da_id = $this->da_id;
        $ec_id = $this->ec_id;
        $h_id = $this->h_id;
        $tmc_id = $this->tmc_id;
        $dc_to_shelter = Path::query()
            ->select(['source_id', 'target_id', 'distance'])
            ->where('path_type', 'ground')
            ->where(function ($query) use ($dc_id, $ec_id) {

                $query->whereIn('source_id', $dc_id)
                    ->WhereIn('target_id', $ec_id);
            })
            ->get()->toArray();
        $da_to_ec = Path::query()
            ->select(['source_id', 'target_id', 'distance'])
            ->where('path_type', 'air')
            ->where(function ($query) use ($da_id, $ec_id) {

                $query->whereIn('source_id', $da_id)
                    ->WhereIn('target_id', $ec_id);
            })
            ->get()->toArray();
        $da_to_h = Path::query()
            ->select(['source_id', 'target_id', 'distance'])
            ->where('path_type', 'ground')
            ->where(function ($query) use ($da_id, $h_id) {

                $query->whereIn('source_id', $da_id)
                    ->WhereIn('target_id', $h_id);
            })
            ->get()->toArray();
        $da_to_h_helicopter = Path::query()
            ->select(['source_id', 'target_id', 'distance'])
            ->where('path_type', 'air')
            ->where(function ($query) use ($da_id, $h_id) {

                $query->whereIn('source_id', $da_id)
                    ->WhereIn('target_id', $h_id);
            })
            ->get()->toArray();
        $da_to_tmc = Path::query()
            ->select(['source_id', 'target_id', 'distance'])
            ->where('path_type', 'ground')
            ->where(function ($query) use ($da_id, $tmc_id) {

                $query->whereIn('source_id', $da_id)
                    ->WhereIn('target_id', $tmc_id);
            })
            ->get()->toArray();
        $da_to_tmc_helicopter = Path::query()
            ->select(['source_id', 'target_id', 'distance'])
            ->where('path_type', 'air')
            ->where(function ($query) use ($da_id, $tmc_id) {

                $query->whereIn('source_id', $da_id)
                    ->WhereIn('target_id', $tmc_id);
            })
            ->get()->toArray();

        $distances =[
            'dc_to_ec' => $dc_to_shelter,
            'da_to_ec' => $da_to_ec,
            'da_to_h' => [
                'ground' => $da_to_h,
                'air' => $da_to_h_helicopter
            ],
            'da_to_tmc' => [
                'ground' => $da_to_tmc,
                'air' => $da_to_tmc_helicopter,
            ]
        ];
        $data = [
            'nodes_id' => [
                'dc' => $dc_id,
                'ec' => $ec_id,
                'da' => $da_id,
                'h' => $h_id,
                'tmc' => $tmc_id,
            ],
            'hsc_parameters' => $this->hsc_parameters[0],
            'affected_pops' => $this->affected_pops,
            'shelters_area' => $this->shelters_area,
            'hospitals_capacity' => $this->hospitals_capacity,
            'tmcs_capacity' => $this->tmcs_capacity,
            'distances' => $distances,
        ];
        return $data;
    }
}