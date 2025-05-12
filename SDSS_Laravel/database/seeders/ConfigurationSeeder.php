<?php

namespace Database\Seeders;

use Illuminate\Database\Console\Seeds\WithoutModelEvents;
use Illuminate\Database\Seeder;
use App\Models\Configuration;
use Illuminate\Support\Facades\DB;
class ConfigurationSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        Configuration::truncate();
        $rows = [
            ['MC1-Standard', 90, 5000, 10, 10 ,10, 100000],
            ['MC2-Lower Budget', 10, 5000, 10, 10 ,10, 100000],
            ['MC3-Lower Safety-Service Level', 90, 5000, 5, 10 ,10, 100000],
            ['MC4-Higher Safety-Service Level', 90, 5000, 20, 10 ,10, 100000],
            ['MC5-Lower Demand Satisfaction', 90, 5000, 10, 10 ,10, 100000]
        ];
        foreach($rows as $row){
            DB::table('configurations')->insert([
                'Name' => $row[0],
                'NT' => $row[1],
                'L' => $row[2],
                'A' => $row[3],
                'G' => $row[4],
                'O' => $row[5],
                'V' => $row[6],
            ]);
        }
    }
}
