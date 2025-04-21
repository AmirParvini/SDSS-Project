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

use function PHPSTORM_META\type;

class SolvingController extends Controller
{
    public function index(Request $request)
    {
        try {
            $APR = $request->input('APR');
            $PP = $request->input('PP');
            $ConfigName = $request->input('Config');

            // read data from Tables-------------------------------------------------------
            $config_parameters = DB::table('configurations')->where('Name', $ConfigName)->first();
            $nodes_data = Node::all()->toArray();
            $commodity_demands = DB::table('scenarios')->where('ArrivalTime(h)', $PP)->first();
            $combinedJson = json_encode([
                'config_parameters' => $config_parameters,
                'nodes_data' => $nodes_data,
                'commodity_demands' => $commodity_demands,
                'APR' => $APR
            ], JSON_UNESCAPED_UNICODE);
            // read data from Tables-------------------------------------------------------

            $pythonScriptPath = base_path('public/python/genetic.py');
            $process = new Process(['python', $pythonScriptPath]);
            $process->setInput($combinedJson);
            $process->setTimeout(900);
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
            $output = json_decode($output, true);
            return response()->json(['status' => 'success', 'output' => $output]);
            // $output = ['test' => 'hello'];
            // return response()->json(['csrf_token' => csrf_token()]);
        } catch (Exception $e) {
            return response()->json([
                'status' => 'error',
                'message' => 'An error occurred: ' . $e->getMessage(),
                'error' => isset($process) ? $process->getErrorOutput() : 'No process output available'
            ], 500);
        }
    }
}
