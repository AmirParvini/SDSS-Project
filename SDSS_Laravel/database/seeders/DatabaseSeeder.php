<?php

namespace Database\Seeders;

use App\Models\Configuration;
use App\Models\Scenario;
// use Illuminate\Database\Console\Seeds\WithoutModelEvents;
use Illuminate\Database\Seeder;
use Illuminate\Support\Facades\DB;

class DatabaseSeeder extends Seeder
{
    /**
     * Seed the application's database.
     */
    public function run(): void
    {
        // User::factory(10)->create();

        // User::factory()->create([
        //     'name' => 'Test User',
        //     'email' => 'test@example.com',
        // ]);

        //initialize onfigurations table----------------------------------------------------------
        Configuration::truncate();
        $rows = [
            ['MC1-Standard', 900, 100, 0.975, 1000000, 5000, 10, 10, 15, 120000, 120000],
            ['MC2-Lower Budget', 100, 10, 0.975, 1000000, 5000, 10, 10, 15, 120000, 120000],
            ['MC3-Lower Safety-Service Level', 900, 100, 0.950, 1000000, 5000, 5, 5, 15, 120000, 120000],
            ['MC4-Higher Safety-Service Level', 900, 100, 0.990, 1000000, 5000, 20, 20, 15, 120000, 120000],
            ['MC5-Lower Demand Satisfaction', 900, 100, 0.975, 1000000, 5000, 10, 10, 1, 120000, 120000]
        ];
        foreach($rows as $row){
            DB::table('configurations')->insert([
                'Name' => $row[0],
                'NT' => $row[1],
                'ni' => $row[2],
                'ST' => $row[3],
                'M' => $row[4],
                'L' => $row[5],
                'A' => $row[6],
                'B' => $row[7],
                'G' => $row[8],
                'V' => $row[9],
                'W' => $row[10],
            ]);
        }


        //initialize scenarios table----------------------------------------------------------
        Scenario::truncate();
        $rows = [
            ['S1', 72, 7.500, 9.0, 1.0, 0.1],
            ['S2', 64, 5.000, 6.0, 0.667, 0.2],
            ['S3', 24, 2.500, 3.0, 0.333, 0.4],
            ['S4', 16, 1.667, 2.0, 0.222, 0.2],
            ['S5', 8, 0.833, 1.0, 0.111, 0.1]
        ];
        foreach ($rows as $row) {
            DB::table('scenarios')->insert([
                'Name' => $row[0],
                'ArrivalTime(h)' => $row[1],
                'Water(unit-pp)' => $row[2],
                'Food(unit-pp)' => $row[3],
                'MedicalKit(unit-pp)' => $row[4],
                'ScenarioProbability' => $row[5],
            ]);
        }


        //initial nodes Table----------------------------------------------------------
        $txtFile = file_get_contents("C:\Users\Amir\Desktop\SDSS-Project\Data\Neighborhood_Data.txt");
        $records = explode("\n", $txtFile);
        $records = array_slice($records, 2);
        $dataToInsert = [];
        foreach ($records as $record) {
            $fields = explode(",", $record);
            $dataToInsert[] = [
                'Neighborhood' => $fields[1],
                'District' => $fields[2][0],
                'City' => 'تهران',
                'NodePopulation' => $fields[4],
                'NodeSaftyLevel' => $fields[5],
                'NodeFacalities' => $fields[6],
                'XCoordinate' => $fields[7],
                'YCoordinate' => $fields[8],
            ];
        }
        // dd($dataToInsert);
        // $chunks = array_chunk($dataToInsert,  1);
        try {
            foreach ($dataToInsert as $row) {
                DB::table('nodes')->insert($row);
            }
            echo "Data inserted successfully.";
        } catch (\Exception $e) {
            echo "Error: " . $e->getMessage();
        }
    }
}
