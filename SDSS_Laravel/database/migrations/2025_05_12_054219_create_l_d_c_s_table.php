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
        Schema::create('l_d_c_s', function (Blueprint $table) {
            $table->id();
            $table->text('Name');
            $table->char('City');
            $table->integer('District');
            $table->double('X');
            $table->double('Y');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('l_d_c_s');
    }
};
