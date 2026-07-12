<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::create('package_flows', function (Blueprint $table) {
            $table->id();
            $table->integer('scenario_id');
            $table->integer('solution_id');
            $table->integer('source_id');
            $table->integer('target_id');
            $table->integer('flow');
            $table->decimal('flow_cost', 10, 2);
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('package_flows');
    }
};
