<?php

namespace Database\Seeders;

use Illuminate\Database\Console\Seeds\WithoutModelEvents;
use Illuminate\Database\Seeder;
use App\Models\LDC;
use Illuminate\Support\Facades\DB;

class LDCSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        LDC::truncate();
        $lines = file("C:/Users/Amir/Desktop/SDSS-Project/Data/LDC_Points.txt");
        $lines = array_slice($lines, 1);
        foreach($lines as $line) {
            $fields = explode(",", $line);
            $dataToInsert[] = [
                'Name' => $fields[5],
                'District' => 4,
                'City' => 'تهران',
                'X' => $fields[count($fields) - 2],
                'Y' => $fields[count($fields) -1]
            ];
        }
        try {
            foreach ($dataToInsert as $row) {
                DB::table('l_d_c_s')->insert($row);
            }
            echo "Data inserted successfully.";
        } catch (\Exception $e) {
            echo "Error: " . $e->getMessage();
        }
    }
}
