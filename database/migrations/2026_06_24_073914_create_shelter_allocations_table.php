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
        Schema::create('shelter_allocations', function (Blueprint $table) {
            $table->id();
            $table->integer('solution_id');
            $table->integer('source_id');
            $table->integer('target_id');
            $table->integer('flow');
            $table->decimal('distance', 10, 2);
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('shelter_allocations');
    }
};
