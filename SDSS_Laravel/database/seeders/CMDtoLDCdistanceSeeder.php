<?php

namespace Database\Seeders;

use App\Models\CMDtoLDCdistance;
use Illuminate\Database\Console\Seeds\WithoutModelEvents;
use Illuminate\Database\Seeder;
use Illuminate\Support\Facades\DB;

class CMDtoLDCdistanceSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        CMDtoLDCdistance::truncate();
        $lines = file("C:/Users/Amir/Desktop/SDSS-Project/Data/OD_Matrix(CMDs_to_LDCs).txt");
        $lines = array_slice($lines, 1);
        $dataToInsert = [];
        foreach ($lines as $line) {
            $fields = explode(",", $line);
            $dataToInsert[] = [
                "CMD_name" => explode("-", $fields[1])[0],
                "LDC_name" => explode("-", $fields[1])[1],
                'Distance' => $fields[count($fields) -1]
            ];
        }
        try {
            foreach ($dataToInsert as $row) {
                DB::table('c_m_dto_l_d_cdistances')->insert($row);
            }
            echo "Data inserted successfully.";
        } catch (\Exception $e) {
            echo "Error: " . $e->getMessage();
        }
    }
}
