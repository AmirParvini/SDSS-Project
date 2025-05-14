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
        Schema::create('l_d_cto_e_cdistances', function (Blueprint $table) {
            $table->id();
            $table->char("City");
            $table->integer("District");
            $table->text("LDC_name");
            $table->text("EC_name");
            $table->double("Distance");
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('l_d_cto_e_cdistances');
    }
};
