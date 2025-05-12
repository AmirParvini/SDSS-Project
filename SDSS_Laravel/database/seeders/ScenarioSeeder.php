<?php

namespace Database\Seeders;

use Illuminate\Database\Console\Seeds\WithoutModelEvents;
use Illuminate\Database\Seeder;
use Illuminate\Support\Facades\DB;

class ScenarioSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        $rows = [
            ["S1", 72, 7.500, 9.0, 1.000, 0.2],
            ["S2", 48, 5.000, 6.0, 0.667, 0.2],
            ["S3", 24, 2.5, 3.0, 0.333, 0.2],
            ["S4", 16, 1.667, 2.0, 0.222, 0.2],
            ["S5", 8, 0.833, 1.0, 0.111, 0.2]
        ];

        foreach ($rows as $row) {
            DB::table('scenarios')->insert([
                'Name' => $row[0],
                'ArrivalTime(h)' => $row[1],
                'Water(unit-pp)' => $row[2],
                'Food(unit-pp)' => $row[3],
                'MedicalKit(unit-pp)' => $row[4],
                'Shelter(unit-pp)' => $row[5],
            ]);
        }
    }
}
