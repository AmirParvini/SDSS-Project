<?php

namespace Database\Seeders;

use Illuminate\Database\Console\Seeds\WithoutModelEvents;
use Illuminate\Database\Seeder;
use App\Models\EC;
use Illuminate\Support\Facades\DB;

class ECSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        EC::truncate();
        $pop = [ 8094.,  1126., 23660., 11591., 18079.,  8744.,    90.,  1406.,
           0.,  8056.,  1744.,  4774., 29267.,  4778.,  3637.,  4777.,
        7210.,  4501.,  4835.,     0.,   887.,   745., 11224.,  1599.,
        2370.,  2645.,  1175.,  2576.,   621.,  8254.,   971.,  4251.,
        4694.,  1862.,   374.,  4131.,   339., 15255.,  5714.,   329.,
        2650., 10277.,  4320., 22302.,  6023.,  6583.,  9966.,  1562.,
        1147., 10057.,  3068.,  1600.,  3220.,  4949.,  7526.,  1752.,
        1089.,   382., 25468.,  2646.,   250.,  3325.,  3782.,  3075.,
        9967., 18778.,  8093.,  7210.,  4896.,  7588.,  4869.,  9521.,
        2983.,   828.,  1884.,   197.,  4616., 14090.,  2388.,  7159.,
        5167., 15596.,   735., 11244.,  1931.,  7066.,  5167.,  5375.,
        2663.,  7651.,  6538., 12019.,  8100.,  1489.,  3596.,  3152.,
        6143.,  6949.,  5324.,  4818.,  2190.,   996., 23405.,  5521.,
        1220., 11170.,  5809., 12502.,  7719.,   954.,  2397.,  1949.,
        6310., 10988.,  8902.,  2617.,  3648.,  4624.,  2224.,  2185.,
         331.,  1311.,  1129.,  1099.,  8394.,  3162.,  4779.,  4221.,
       25823.,  4777.,  4983.,  7849.,  4267.,  4371.,  3396.,  3152.,
        4090.,  7791.,  2182.,  8832.,  9601.,  7652.,   873.,  3186.,
        6460.,  6573., 10798.,  2933.,  7613.,  2819.,  6106., 12982.,
        1773.,   504.,  4001., 10390.,  2712.,  7159.];
        $lines = file("C:/Users/Amir/Desktop/SDSS-Project/Data/EC_Points.txt");
        $lines = array_slice($lines, 1);
        $dataToInsert = [];
        foreach ($lines as $index => $line) {
            $fields = explode(",", $line);
            $dataToInsert[] = [
                'Name' => $fields[2],
                'District' => 4,
                'City' => 'تهران',
                'X' => $fields[count($fields) - 2],
                'Y' => $fields[count($fields) -1],
                'Population' => $pop[$index]
            ];
        }
        try {
            foreach ($dataToInsert as $row) {
                DB::table('e_c_s')->insert($row);
            }
            echo "Data inserted successfully.";
        } catch (\Exception $e) {
            echo "Error: " . $e->getMessage();
        }
    }
}
