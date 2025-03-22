<?php

namespace Database\Seeders;

use App\Models\Configuration;
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

        Configuration::truncate();
        DB::table('configurations')->insert([
            'Name' => 'standard',
            'NT' => 400,
            'ST' => 0.975,
            'M' => 1000000,
            'L' => 5000,
            'A' => 10,
            'B' => 10,
            'G' => 15,
            'V' => 150000,
            'W' => 150000,
        ]);



        // For nodes Table
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
