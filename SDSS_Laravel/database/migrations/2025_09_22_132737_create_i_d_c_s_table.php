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
        Schema::create('i_d_c_s', function (Blueprint $table) {
            $table->id();
            $table->string('name');
            $table->string('capacity');
            $table->string('coordinate');
            $table->string('fixed_cost');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('i_d_c_s');
    }
};
