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
        Schema::create('da_ec_path', function (Blueprint $table) {
            $table->id();
            $table->integer('da_id');
            $table->integer('ec_id');
            $table->float('distance');
            $table->text('path');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('da_ec_path');
    }
};
