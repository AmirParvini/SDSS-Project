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
        Schema::table('damaged_areas', function (Blueprint $table) {
            $table->dropColumn('injured');
            $table->integer('affected_pop')->after('name');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('damaged_areas', function (Blueprint $table) {
            $table->dropColumn('injured');
            $table->integer('affected_pop');
        });
    }
};
