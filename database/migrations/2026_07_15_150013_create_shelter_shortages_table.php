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
        Schema::create('shelter_shortages', function (Blueprint $table) {
            $table->id();
            $table->foreignId("scenario_id")->constrained();
            $table->integer("solution_id");
            $table->foreignId("node_id")->constrained();
            $table->integer("shortage");
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('shelter_shortages');
    }
};
