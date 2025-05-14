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
        Schema::create('c_m_dto_l_d_cdistances', function (Blueprint $table) {
            $table->id();
            $table->char("City");
            $table->integer("District");
            $table->text("CMD_name");
            $table->text("LDC_name");
            $table->double("Distance");
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('c_m_dto_l_d_cdistances');
    }
};
