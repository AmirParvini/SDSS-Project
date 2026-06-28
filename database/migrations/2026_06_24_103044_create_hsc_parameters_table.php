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
        Schema::create('hsc_parameters', function (Blueprint $table) {
            $table->id();
            $table->integer('scenario_id');
            $table->decimal('budget', 10, 2);
            $table->decimal('t1', 10, 3);
            $table->decimal('t2', 10, 3);
            $table->decimal('pua', 10, 2);
            $table->decimal('rta', 10, 2);
            $table->integer('rtc');
            $table->integer('gv_speed');
            $table->integer('gv_severe_capacity');
            $table->integer('gv_moderate_capacity');
            $table->integer('av_speed');
            $table->integer('av_severe_capacity');
            $table->integer('av_moderate_capacity');
            $table->decimal('phi_min_s', 10, 2);
            $table->decimal('phi_max_s', 10, 2);
            $table->decimal('ks_s', 10, 2);
            $table->integer('tm_s');
            $table->decimal('phi_min_m', 10, 2);
            $table->decimal('phi_max_m', 10, 2);
            $table->decimal('ks_m', 10, 2);
            $table->integer('tm_m');
            $table->integer('itst');
            $table->integer('wt');
            $table->decimal('rp_cost', 10, 2);
            $table->decimal('rpt_cost', 10, 2);
            $table->decimal('tmc_cost', 10, 2);
            $table->decimal('shelter_cost', 10, 2);
            $table->decimal('gv_cost', 10, 2);
            $table->decimal('av_cost', 10, 2);
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('hsc_parameters');
    }
};
