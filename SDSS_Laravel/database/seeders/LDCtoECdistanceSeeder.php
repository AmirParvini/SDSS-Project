<?php

namespace Database\Seeders;

use App\Models\LDCtoECdistance;
use Illuminate\Database\Console\Seeds\WithoutModelEvents;
use Illuminate\Database\Seeder;
use Illuminate\Support\Facades\DB;
class LDCtoECdistanceSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        LDCtoECdistance::truncate();
        $lines = file("C:/Users/Amir/Desktop/SDSS-Project/Data/OD_Matrix(LDCs_to_ECs).txt");
        $lines = array_slice($lines, 1);
        $dataToInsert = [];
        foreach ($lines as $line) {
            $fields = explode(",", $line);
            $dataToInsert[] = [
                "LDC_name" => explode("-", $fields[1])[0],
                "EC_name" => explode("-", $fields[1])[1],
                'Distance' => $fields[count($fields) -1]
            ];
        }
        try {
            foreach ($dataToInsert as $row) {
                DB::table('l_d_cto_e_cdistances')->insert($row);
            }
            echo "Data inserted successfully.";
        } catch (\Exception $e) {
            echo "Error: " . $e->getMessage();
        }
    }
}
