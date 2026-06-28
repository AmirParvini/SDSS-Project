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
        Schema::table('affected_areas', function (Blueprint $table) {
            $table->dropColumn('name');
            $table->dropColumn('lat');
            $table->dropColumn('lng');
            $table->dropTimestamps();
            $table->renameColumn('id', 'node_id');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        // Schema::table('affected_areas', function (Blueprint $table) {
        //     $table->dropColumn('injured');
        //     $table->integer('affected_pop');
        // });
    }
};
