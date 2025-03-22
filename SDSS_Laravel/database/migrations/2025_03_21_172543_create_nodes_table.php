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
        Schema::create('nodes', function (Blueprint $table) {
            $table->id();
            $table->integer('Region')->nullable();
            $table->char('City')->nullable();
            $table->char('District')->nullable();
            $table->char('Neighborhood')->nullable();
            $table->float('NodeSaftyLevel')->nullable();
            $table->integer('NodePopulation')->nullable();
            $table->integer('NodeFacalities')->nullable();
            $table->double('NodeArea')->nullable();
            $table->double('XCoordinate')->nullable();
            $table->double('YCoordinate')->nullable();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('nodes');
    }
};
