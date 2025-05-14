<?php

namespace App\Http\Controllers;

use App\Models\Node;
use Exception;
use Illuminate\Http\Client\Request as ClientRequest;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Http;
use Symfony\Component\Process\Process;
use Symfony\Component\Process\Exception\ProcessFailedException;
use App\Models\CMD;
use App\Models\CMDtoLDCdistance;
use App\Models\LDC;
use App\Models\EC;
use App\Models\LDCtoECdistance;

use function PHPSTORM_META\type;

class SolvingController extends Controller
{
    public function index(Request $request)
    {
        try {
            $City = $request->input('City');
            $District = $request->input('District');
            $APR = $request->input('APR');
            $PP = $request->input('PP');
            $ConfigName = $request->input('Config');
            // $APR = 0.9;
            // $PP = 8;
            // $ConfigName = "MC1-Standard";

            // read data from Tables-------------------------------------------------------
            $config_parameters = DB::table('configurations')->where('Name', $ConfigName)->first();
            $CMD_nodes_data = CMD::all()->toArray();
            $LDC_nodes_data = LDC::all()->toArray();
            $EC_nodes_data = EC::all()->toArray();
            $CMD_to_LDC_Distances_data = DB::table('c_m_dto_l_d_cdistances')
                ->where('City', $City)
                ->where('District', $District)->get();
            $LDC_to_EC_Distances_data = DB::table('l_d_cto_e_cdistances')
                ->where('City', $City)
                ->where('District', $District)->get();
            $nodes_data = [$CMD_nodes_data, $LDC_nodes_data, $EC_nodes_data];
            $nodes_dist_data = [$CMD_to_LDC_Distances_data, $LDC_to_EC_Distances_data];
            $commodity_demands_unit = DB::table('scenarios')->where('ArrivalTime(h)', $PP)->first();
            $combinedJson = json_encode([
                'config_parameters' => $config_parameters,
                'nodes_data' => $nodes_data,
                'nodes_dist_data' => $nodes_dist_data,
                'commodity_demands_unit' => $commodity_demands_unit,
                'APR' => $APR
            ], JSON_UNESCAPED_UNICODE);
            // read data from Tables-------------------------------------------------------

            $pythonScriptPath = base_path('public/python/genetic_NewModel.py');
            $process = new Process(['python', $pythonScriptPath]);
            $process->setInput($combinedJson);
            $process->setTimeout(1800);
            $process->mustRun();
            if (!$process->isSuccessful()) {
                // throw new ProcessFailedException($process);
                return response()->json([
                    'status' => 'error',
                    'message' => 'Process failed',
                    'error' => $process->getErrorOutput()
                ], 500);
            }
            $output = $process->getOutput();
            dd($output);
            $output = json_decode($output, true);
            return response()->json(['status' => 'success', 'output' => $output]);
        } catch (Exception $e) {
            return response()->json([
                'status' => 'error',
                'message' => 'An error occurred = ' . $e->getMessage(),
                // 'error' => isset($process) ? $process->getErrorOutput() : 'No process output available'
            ], 500);
        }
    }
}
