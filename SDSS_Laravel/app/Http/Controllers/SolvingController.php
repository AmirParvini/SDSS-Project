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
    var $record;
    public function index(Request $request)
    {
        set_time_limit(120);
        // read data from Tables
        $record = DB::table('configurations')->where('Name', 'standard')->first();
        $parameters =  $record;
        $record = Node::all()->toArray();
        $nodes_data =  $record;
        $combinedJson = json_encode(['parameters' => $parameters, 'nodes_data' => $nodes_data], JSON_UNESCAPED_UNICODE);
        // read data from Tables
        try {
            $pythonScriptPath = base_path('public/python/genetic.py');
            $process = new Process(['python', $pythonScriptPath]);
            $process->setInput($combinedJson);
            $process->mustRun();
            if (!$process->isSuccessful()) {
                throw new ProcessFailedException($process);
            }
            $output = $process->getOutput();
            return $output;
        } catch (Exception $e) {
            dd($e);
        }
    }
}
