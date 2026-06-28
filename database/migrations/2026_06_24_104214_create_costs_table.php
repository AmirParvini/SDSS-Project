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
        Schema::create('costs', function (Blueprint $table) {
            $table->id();
            $table->integer('solution_id');
            $table->decimal('package_flow_cost', 10, 2);
            $table->decimal('package_cost', 10, 2);
            $table->decimal('ground_vehicle_cost', 10, 2);
            $table->decimal('air_vehicle_cost', 10, 2);
            $table->decimal('shelter_establish_cost', 10, 2);
            $table->decimal('tmc_establish_cost', 10, 2);
            $table->decimal('total_cost', 10, 2);
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('costs');
    }
};
