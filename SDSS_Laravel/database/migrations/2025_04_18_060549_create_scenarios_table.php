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
        Schema::create('scenarios', function (Blueprint $table) {
            $table->id();
            $table->char('Name')->nullable();
            $table->integer('ArrivalTime(h)')->nullable();
            $table->float('Water(unit-pp)')->nullable();
            $table->float('Food(unit-pp)')->nullable();
            $table->float('MedicalKit(unit-pp)')->nullable();
            $table->float('ScenarioProbability')->nullable();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('scenarios');
    }
};
