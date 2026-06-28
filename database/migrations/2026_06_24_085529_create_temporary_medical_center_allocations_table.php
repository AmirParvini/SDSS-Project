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
        Schema::create('temporary_medical_center_allocations', function (Blueprint $table) {
            $table->id();
            $table->string('name');
            $table->integer('solution_id');
            $table->integer('source_id');
            $table->integer('target_id');
            $table->integer('g_flow_moderate');
            $table->integer('num_gv_moderate');
            $table->decimal('g_flow_cost_moderate', 10, 2);
            $table->integer('a_flow_moderate');
            $table->integer('num_av_moderate');
            $table->decimal('a_flow_cost_moderate', 10, 2);
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('temporary_medical_center_allocations');
    }
};
