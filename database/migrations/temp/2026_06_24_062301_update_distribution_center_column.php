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
        Schema::table('distribution_centers', function (Blueprint $table) {
            // $table->dropColumn('name');
            // $table->dropColumn('lat');
            // $table->dropColumn('lng');
            // $table->dropColumn('fixed_cost');
            $table->dropColumn('capacity');
            // $table->dropTimestamps();
            // $table->renameColumn('id', 'node_id');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        //
    }
};
