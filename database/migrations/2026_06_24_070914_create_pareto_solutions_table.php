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
        Schema::create('pareto_solutions', function (Blueprint $table) {
            $table->id();
            $table->integer('scenario_id');
            $table->decimal('z1', 10, 2);
            $table->decimal('z2', 10, 2);
            $table->decimal('z3', 10, 2);
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('pareto_solutions');
    }
};
