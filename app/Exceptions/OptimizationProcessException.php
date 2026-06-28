<?php

namespace App\Exceptions;

use RuntimeException;
use Symfony\Component\Process\Process;
use Throwable;

class OptimizationProcessException extends RuntimeException
{
    public function __construct(
        string $message,
        protected readonly string $errorOutput = '',
        protected readonly ?int $exitCode = null,
        ?Throwable $previous = null,
    ) {
        parent::__construct($message, 0, $previous);
    }

    /**
     * ساخت Exception مستقیماً از روی یک Process ناموفق.
     * این متد جای تکرار منطق استخراج خطا در RunProcess را می‌گیرد.
     */
    public static function fromFailedProcess(Process $process): self
    {
        return new self(
            message: 'Optimization process failed to execute.',
            errorOutput: $process->getErrorOutput(),
            exitCode: $process->getExitCode(),
        );
    }

    public static function invalidOutput(string $rawOutput): self
    {
        return new self(
            message: 'Optimization process returned invalid JSON output.',
            errorOutput: $rawOutput,
        );
    }

    public function getErrorOutput(): string
    {
        return $this->errorOutput;
    }

    public function getExitCode(): ?int
    {
        return $this->exitCode;
    }
}