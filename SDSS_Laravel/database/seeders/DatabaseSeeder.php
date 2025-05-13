<?php

namespace Database\Seeders;

use App\Models\CMD;
use App\Models\CMD_to_LDC_distance;
use App\Models\CMDtoLDCdistance;
use App\Models\LDC_to_EC_distance;
use App\Models\Configuration;
use App\Models\EC;
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

        $this->call([
            ConfigurationSeeder::class,
            CMDSeeder::class,
            LDCSeeder::class,
            ECSeeder::class,
            ScenarioSeeder::class,
            CMDtoLDCdistanceSeeder::class,
            LDCtoECdistanceSeeder::class
        ]);


        //initialize scenarios table----------------------------------------------------------



        //initial nodes Table----------------------------------------------------------
        // $txtFile = file_get_contents("C:\Users\Amir\Desktop\SDSS-Project\Data\Neighborhood_Data.txt");
        // $records = explode("\n", $txtFile);
        // $records = array_slice($records, 2);
        // $dataToInsert = [];
        // foreach ($records as $record) {
        //     $fields = explode(",", $record);
        //     $dataToInsert[] = [
        //         'Neighborhood' => $fields[1],
        //         'District' => $fields[2][0],
        //         'City' => 'تهران',
        //         'NodePopulation' => $fields[4],
        //         'NodeSaftyLevel' => $fields[5],
        //         'NodeFacalities' => $fields[6],
        //         'XCoordinate' => $fields[7],
        //         'YCoordinate' => $fields[8],
        //     ];
        // }
        // // dd($dataToInsert);
        // // $chunks = array_chunk($dataToInsert,  1);
        // try {
        //     foreach ($dataToInsert as $row) {
        //         DB::table('nodes')->insert($row);
        //     }
        //     echo "Data inserted successfully.";
        // } catch (\Exception $e) {
        //     echo "Error: " . $e->getMessage();
        // }
    }
}
