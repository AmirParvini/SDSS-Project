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
        Schema::create('configurations', function (Blueprint $table) {
            $table->id()->autoIncrement();
            $table->char('name');
            $table->integer('NT');
            $table->float('ST');
            $table->integer('M');
            $table->integer('L');
            $table->integer('A');
            $table->integer('B');
            $table->integer('G');
            $table->integer('V');
            $table->integer('W');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('configurations');
    }
};
