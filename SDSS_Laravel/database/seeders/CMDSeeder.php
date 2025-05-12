<?php

namespace Database\Seeders;

use Illuminate\Database\Console\Seeds\WithoutModelEvents;
use Illuminate\Database\Seeder;
use App\Models\CMD;
use Illuminate\Support\Facades\DB;

class CMDSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        CMD::truncate();
        $lines = file("C:/Users/Amir/Desktop/SDSS-Project/Data/CMD_Points.txt");
        $lines = array_slice($lines, 1);
        $dataToInsert = [];
        foreach ($lines as $line) {
            $fields = explode(",", $line);
            $dataToInsert[] = [
                'Name' => $fields[1],
                'District' => 4,
                'City' => 'تهران',
                'X' => $fields[count($fields) - 2],
                'Y' => $fields[count($fields) -1]
            ];
        }
        try {
            foreach ($dataToInsert as $row) {
                DB::table('c_m_d_s')->insert($row);
            }
            echo "Data inserted successfully.";
        } catch (\Exception $e) {
            echo "Error: " . $e->getMessage();
        }
    }
}
